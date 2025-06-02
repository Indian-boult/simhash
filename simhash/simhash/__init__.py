# Created by 1e0n in 2013
from __future__ import division, unicode_literals

import re
import sys
import hashlib
import logging
import numbers
import collections.abc
from collections.abc import Iterable
from itertools import groupby
import numpy as np

if sys.version_info[0] >= 3:
    basestring = str
    unicode = str
    long = int

    def int_to_bytes(n, length):
        return n.to_bytes(length, 'big')

    def bytes_to_int(b):
        return int.from_bytes(b, 'big')
else:
    range = xrange

    def int_to_bytes(n, length):
        return '{:0{}x}'.format(n, length * 2).decode('hex')

    def bytes_to_int(b):
        return int(b.encode('hex'), 16)

def _hashfunc(x):
    """Default hash function using MD5.
    
    Returns the digest bytes (not an integer).
    """
    return hashlib.md5(x).digest()


class Simhash(object):
    # Constants used in calculating simhash. Larger values will use more RAM.
    large_weight_cutoff = 50
    batch_size = 200

    def __init__(
            self, value, f=64, reg=r'[\w\u4e00-\u9fcc]+', hashfunc=_hashfunc, log=None
    ):
        """
        `f` is the dimensions of fingerprints, in bits. Must be a multiple of 8.

        `reg` is meaningful only when `value` is basestring and describes
        what is considered to be a letter inside parsed string. Regexp
        object can also be specified (some attempt to handle any letters
        is to specify reg=re.compile(r'\\w', re.UNICODE))

        `hashfunc` accepts a utf-8 encoded string and returns either bytes
        (preferred) or an unsigned integer, in at least `f // 8` bytes.
        """
        if f % 8:
            raise ValueError('f must be a multiple of 8')

        self.f = f
        self.f_bytes = f // 8  # Number of bytes needed for f bits
        self.reg = reg
        self.value = None
        
        # Setup hash function and detect return type
        self.hashfunc = hashfunc
        test_hash = hashfunc(b"test")
        self.hashfunc_returns_int = isinstance(test_hash, numbers.Integral)

        if log is None:
            self.log = logging.getLogger("simhash")
        else:
            self.log = log

        # Process the input value
        if isinstance(value, Simhash):
            self.value = value.value
        elif isinstance(value, basestring):
            self.build_by_text(unicode(value))
        elif isinstance(value, Iterable):
            self.build_by_features(value)
        elif isinstance(value, numbers.Integral):
            self.value = value & ((1 << self.f) - 1)  # Ensure value fits within f bits
        else:
            raise Exception('Bad parameter with type {}'.format(type(value)))

    def __eq__(self, other):
        """
        Compare two simhashes by their value.

        :param Simhash other: The Simhash object to compare to
        """
        return self.value == other.value

    def _slide(self, content, width=4):
        """Generate sliding windows of the content with specified width."""
        return [content[i:i + width] for i in range(max(len(content) - width + 1, 1))]

    def _tokenize(self, content):
        """Tokenize the content string into features.
        
        By default, extracts words according to self.reg pattern and creates n-grams.
        """
        content = unicode(content).lower()
        if isinstance(self.reg, basestring):
            reg = re.compile(self.reg)
        else:
            reg = self.reg
        
        # Extract words
        words = reg.findall(content)
        
        # Generate n-grams (sliding windows)
        features = []
        for word in words:
            features.extend(self._slide(word))
            
        return features

    def build_by_text(self, content):
        """Build the simhash from a text string."""
        features = self._tokenize(content)
        features = {k:sum(1 for _ in g) for k, g in groupby(sorted(features))}
        return self.build_by_features(features)

    def build_by_features(self, features):
        """
        Build a simhash value from features.
        
        `features` might be:
        - a list of unweighted tokens (a weight of 1 will be assumed)
        - a list of (token, weight) tuples
        - a token -> weight dict
        
        This implementation processes features in batches for memory efficiency,
        and optimizes handling for large feature sets and weights.
        """
        # Use numpy for efficient computation
        v = np.zeros(self.f, dtype=np.float64)
        
        # Convert dict to items if necessary
        if isinstance(features, dict):
            features = list(features.items())
        
        # No features provided
        if not features:
            self.value = 0
            return self

        # Process features in batches for memory efficiency
        feature_count = 0
        batch = []
        weights = []
        
        # Convert features to a proper list if it's an iterator
        if not isinstance(features, list):
            features = list(features)
        
        for feature in features:
            weight = 1
            
            # Handle weighted features
            if not isinstance(feature, basestring):
                feature, weight = feature
            
            feature_count += 1
            
            # Get the hash for this feature
            if self.hashfunc_returns_int:
                # For integer hash results
                feature_hash = self.hashfunc(feature.encode('utf-8'))
                # Ensure it's within f bits
                feature_hash &= (1 << self.f) - 1
                hash_bits = self._int_to_bits(feature_hash)
            else:
                # For bytes hash results
                hash_bytes = self.hashfunc(feature.encode('utf-8'))
                # Use specific number of bytes
                if len(hash_bytes) >= self.f_bytes:
                    hash_bytes = hash_bytes[:self.f_bytes]
                else:
                    # Pad if too short
                    hash_bytes = hash_bytes.ljust(self.f_bytes, b'\x00')
                
                hash_bits = self._bytes_to_bits(hash_bytes)
            
            # Special handling for features with large weights
            if not isinstance(weight, int) or weight > self.large_weight_cutoff:
                # For large or non-integer weights, we update v directly
                v += np.array(hash_bits) * weight
            else:
                # For normal weights, we batch process
                batch.append(hash_bits)
                weights.append(weight)
                
                # Process batch if it's full
                if len(batch) >= self.batch_size:
                    self._process_batch(v, batch, weights)
                    batch = []
                    weights = []
        
        # Process any remaining batch items
        if batch:
            self._process_batch(v, batch, weights)
        
        # Determine threshold and convert to binary
        threshold = feature_count / 2.0
        binary = np.where(v > threshold, 1, 0)
        
        # Convert binary to integer value
        value = 0
        for bit in binary:
            value = (value << 1) | int(bit)
            
        self.value = value
        return self
    
    def _process_batch(self, v, batch, weights):
        """Process a batch of hashed features with weights efficiently."""
        for i, bits in enumerate(batch):
            weight = weights[i]
            v += np.array(bits) * weight
    
    def _int_to_bits(self, n):
        """Convert an integer to a list of bits (0 or 1)."""
        bits = []
        for i in range(self.f):
            bits.append(1 if (n & (1 << (self.f - 1 - i))) else 0)
        return bits
    
    def _bytes_to_bits(self, hash_bytes):
        """Convert bytes to a list of bits (0 or 1)."""
        bits = []
        for b in hash_bytes:
            if not isinstance(b, int):
                b = ord(b)  # Handle Python 2 string compatibility
            for i in range(8):
                bits.append(1 if (b & (1 << (7 - i))) else 0)
        return bits[:self.f]  # Only use the first f bits

    def distance(self, another):
        """Calculate the Hamming distance between two simhashes."""
        assert self.f == another.f
        # XOR the values and count the number of set bits
        x = (self.value ^ another.value) & ((1 << self.f) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1  # Brian Kernighan's algorithm to count set bits
        return ans


class SimhashIndex(object):
    """
    Index of simhashes for quick search of near-duplicates.
    
    Based on the paper "Detecting Near-Duplicates for Web Crawling" by Manku et al.
    """
    
    def __init__(self, objs, f=64, k=2, log=None):
        """
        `objs` is a list of (obj_id, simhash)
        obj_id is a string, simhash is an instance of Simhash
        `f` is the same with the one for Simhash
        `k` is the tolerance
        """
        self.k = k
        self.f = f
        count = len(objs)

        if log is None:
            self.log = logging.getLogger("simhash")
        else:
            self.log = log

        self.log.info('Initializing %s data.', count)

        self.bucket = collections.defaultdict(set)

        for i, q in enumerate(objs):
            if i % 10000 == 0 or i == count - 1:
                self.log.info('%s/%s', i + 1, count)

            self.add(*q)

    def get_near_dups(self, simhash):
        """
        `simhash` is an instance of Simhash
        return a list of obj_id, which is in type of str
        """
        assert simhash.f == self.f
        
        # Find near-duplicates
        ans = set()
        for key in self.get_keys(simhash):
            dups = self.bucket[key]
            self.log.debug('key:%s', key)
            if len(dups) > 200:
                self.log.warning('Big bucket found. key:%s, len:%s', key, len(dups))

            for dup in dups:
                sim2, obj_id = dup.split(',', 1)
                # Convert hex string to integer and create Simhash object
                sim2_val = int(sim2, 16)
                sim2 = Simhash(sim2_val, self.f)

                d = simhash.distance(sim2)
                if d <= self.k:
                    ans.add(obj_id)
        return list(ans)

    def add(self, obj_id, simhash):
        """
        `obj_id` is a string
        `simhash` is an instance of Simhash
        """
        assert simhash.f == self.f

        for key in self.get_keys(simhash):
            v = '%x,%s' % (simhash.value, obj_id)
            self.bucket[key].add(v)

    def delete(self, obj_id, simhash):
        """
        `obj_id` is a string
        `simhash` is an instance of Simhash
        """
        assert simhash.f == self.f

        for key in self.get_keys(simhash):
            v = '%x,%s' % (simhash.value, obj_id)
            if v in self.bucket[key]:
                self.bucket[key].remove(v)

    @property
    def offsets(self):
        """
        Calculate offsets for the simhash buckets.
        
        This implementation uses the technique described in the paper:
        "Detecting Near-Duplicates for Web Crawling" by Manku et al.
        """
        return [self.f // (self.k + 1) * i for i in range(self.k + 1)]

    def get_keys(self, simhash):
        """Generate keys for the simhash based on bit offsets."""
        for i, offset in enumerate(self.offsets):
            if i == (len(self.offsets) - 1):
                m = 2 ** (self.f - offset) - 1
            else:
                m = 2 ** (self.offsets[i + 1] - offset) - 1
            c = simhash.value >> offset & m
            yield '%x:%x' % (c, i)

    def bucket_size(self):
        """Return the number of buckets in the index."""
        return len(self.bucket)
