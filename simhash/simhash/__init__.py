# Created by 1e0n in 2013
from __future__ import division, unicode_literals

import re
import sys
import hashlib
import logging
import numbers
import collections.abc
from itertools import groupby
import numpy as np

if sys.version_info[0] >= 3:
    basestring = str
    unicode = str
    long = int

    def int_to_bytes(n, length):
        # Handle very large integers by truncating to the desired length
        mask = (1 << (length * 8)) - 1
        n = n & mask
        return n.to_bytes(length, 'big')

    def bytes_to_int(b):
        return int.from_bytes(b, 'big')
else:
    range = xrange

    def int_to_bytes(n, length):
        # Handle very large integers by truncating to the desired length
        mask = (1 << (length * 8)) - 1
        n = n & mask
        return '{:0{}x}'.format(n, length * 2).decode('hex')

    def bytes_to_int(b):
        return int(b.encode('hex'), 16)

def _hashfunc(x):
    """Default hash function using MD5 and returning digest bytes"""
    return hashlib.md5(x).digest()


class Simhash(object):
    # Constants used in calculating simhash. Larger values will use more RAM.
    large_weight_cutoff = 50
    batch_size = 200

    def __init__(
            self, value, f=64, reg=r'[a-zA-Z0-9_\u4e00-\u9fcc]+', hashfunc=_hashfunc, log=None
    ):
        """
        `f` is the dimensions of fingerprints, in bits. Must be a multiple of 8.

        `reg` is meaningful only when `value` is basestring and describes
        what is considered to be a letter inside parsed string. Regexp
        object can also be specified (some attempt to handle any letters
        is to specify reg=re.compile(r'[a-zA-Z0-9_]', re.UNICODE))

        `hashfunc` accepts a utf-8 encoded string and returns either bytes
        (preferred) or an unsigned integer, in at least `f // 8` bytes.
        """
        if f % 8:
            raise ValueError('f must be a multiple of 8')

        self.f = f
        self.f_bytes = f // 8
        self.reg = reg
        self.value = None
        self.hashfunc = hashfunc
        self.hashfunc_returns_int = isinstance(hashfunc(b"test"), numbers.Integral)

        if log is None:
            self.log = logging.getLogger("simhash")
        else:
            self.log = log

        if isinstance(value, Simhash):
            self.value = value.value
        elif isinstance(value, basestring):
            self.build_by_text(unicode(value))
        elif isinstance(value, collections.abc.Iterable):
            self.build_by_features(value)
        elif isinstance(value, numbers.Integral):
            self.value = value
        else:
            raise Exception('Bad parameter with type {}'.format(type(value)))

    def __eq__(self, other):
        """
        Compare two simhashes by their value.

        :param Simhash other: The Simhash object to compare to
        """
        return self.value == other.value

    def _slide(self, content, width=4):
        """
        Slide a window of the specified width through the content to create tokens.
        Ensures proper handling of short strings and edge cases.
        """
        if not content or len(content) <= width:
            return [content] if content else []
        else:
            return [content[i:i + width] for i in range(len(content) - width + 1)]

    def _tokenize(self, content):
        """
        Tokenize a string into features suitable for simhash generation.
        Enhances robustness by preserving word boundaries and handling special cases.
        """
        content = content.lower()
        words = re.findall(self.reg, content)
        
        # Handle short strings: return the full string as a token
        if len(content) < 10 and len(words) <= 1:
            return [content]
        
        # Create character n-grams from content
        tokens = self._slide(''.join(words))
        
        # For languages like Chinese, also use the words themselves as tokens
        if any('\u4e00' <= c <= '\u9fff' for c in content):
            tokens.extend(words)
            
        # Add word-level features for more discriminative power
        if len(words) > 1:
            word_pairs = self._slide(' '.join(words), 2)
            tokens.extend(word_pairs)
            
        return tokens

    def build_by_text(self, content):
        features = self._tokenize(content)
        features = {k: sum(1 for _ in g) for k, g in groupby(sorted(features))}
        return self.build_by_features(features)

    def build_by_features(self, features):
        """
        `features` might be a list of unweighted tokens (a weight of 1
                  will be assumed), a list of (token, weight) tuples or
                  a token -> weight dict.
        """
        # Prepare a bitmask for proper bit length truncation
        mask = (1 << self.f) - 1
        
        v = [0] * self.f  # Initialize v to f zeros (one for each bit position)
        
        # Convert dict to items if needed
        if isinstance(features, dict):
            features = features.items()

        # Process features in batches for better memory efficiency
        feature_batches = []
        current_batch = []
        
        for feature in features:
            # Extract feature and weight
            if isinstance(feature, basestring):
                f, w = feature, 1
            else:
                f, w = feature
                
            # Skip empty features
            if not f:
                continue
                
            # Get the hash for this feature
            if self.hashfunc_returns_int:
                hash_int = self.hashfunc(f.encode('utf-8'))
                # Ensure we only use the lowest bits needed
                hash_int &= mask
            else:
                # Get bytes and convert to integer
                hash_bytes = self.hashfunc(f.encode('utf-8'))
                # Use the last self.f_bytes of the hash
                hash_bytes = hash_bytes[-self.f_bytes:]
                hash_int = bytes_to_int(hash_bytes) & mask
                
            # If the weight is very large or non-integer, handle directly
            if w > self.large_weight_cutoff or isinstance(w, float):
                # For large weights, we'll process directly rather than batching
                for i in range(self.f):
                    bitmask = 1 << i
                    if hash_int & bitmask:
                        v[i] += w
                    else:
                        v[i] -= w
            else:
                # For reasonable integer weights, add to batch
                if w == 1:
                    current_batch.append(hash_int)
                else:
                    # Convert to int to handle numpy types
                    w_int = int(w)
                    current_batch.extend([hash_int] * w_int)
                    
            # Process batch if it gets too large
            if len(current_batch) >= self.batch_size:
                feature_batches.append(current_batch)
                current_batch = []
        
        # Don't forget the last batch
        if current_batch:
            feature_batches.append(current_batch)
        
        # Process all batches
        for batch in feature_batches:
            for hash_int in batch:
                for i in range(self.f):
                    bitmask = 1 << i
                    if hash_int & bitmask:
                        v[i] += 1
                    else:
                        v[i] -= 1
        
        # Finalize the simhash value
        simhash = 0
        for i in range(self.f):
            if v[i] > 0:
                simhash |= 1 << i
                
        self.value = simhash
        return self

    def distance(self, another):
        """
        Calculate the Hamming distance between two simhashes.
        
        Args:
            another: Another Simhash object to compare with
            
        Returns:
            The hamming distance (number of different bits)
        """
        assert self.f == another.f
        
        # XOR the two values to get bits that differ
        x = (self.value ^ another.value) & ((1 << self.f) - 1)
        
        # Count the number of set bits using Brian Kernighan's algorithm
        # This is more efficient than naive bit counting
        ans = 0
        while x:
            ans += 1
            x &= x - 1  # Clear the least significant bit set
            
        return ans


class SimhashIndex(object):

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

        ans = set()

        for key in self.get_keys(simhash):
            dups = self.bucket[key]
            self.log.debug('key:%s', key)
            if len(dups) > 200:
                self.log.warning('Big bucket found. key:%s, len:%s', key, len(dups))

            for dup in dups:
                sim2, obj_id = dup.split(',', 1)
                sim2 = Simhash(long(sim2, 16), self.f)

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
        You may optimize this method according to <http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33026.pdf>
        """
        return [self.f // (self.k + 1) * i for i in range(self.k + 1)]

    def get_keys(self, simhash):
        for i, offset in enumerate(self.offsets):
            if i == (len(self.offsets) - 1):
                m = 2 ** (self.f - offset) - 1
            else:
                m = 2 ** (self.offsets[i + 1] - offset) - 1
            c = simhash.value >> offset & m
            yield '%x:%x' % (c, i)

    def bucket_size(self):
        return len(self.bucket)
