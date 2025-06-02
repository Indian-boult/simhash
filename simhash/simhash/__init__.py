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
        """Convert integer to bytes with specific length"""
        try:
            return n.to_bytes(length, 'big')
        except OverflowError:
            # Handle overflow by masking to the required number of bits
            mask = (1 << (length * 8)) - 1
            return (n & mask).to_bytes(length, 'big')

    def bytes_to_int(b):
        """Convert bytes to integer"""
        return int.from_bytes(b, 'big')
else:
    range = xrange

    def int_to_bytes(n, length):
        """Convert integer to bytes with specific length for Python 2"""
        mask = (1 << (length * 8)) - 1
        n = n & mask
        return '{:0{}x}'.format(n, length * 2).decode('hex')

    def bytes_to_int(b):
        """Convert bytes to integer for Python 2"""
        return int(b.encode('hex'), 16)

def _hashfunc(x):
    """
    Default hash function using MD5 and returning digest bytes.
    This is more consistent than returning an integer representation.
    """
    return hashlib.md5(x).digest()


class Simhash(object):
    """
    Simhash implementation with optimizations for memory efficiency
    and support for large feature sets.
    """
    # Constants used in calculating simhash. Larger values will use more RAM.
    large_weight_cutoff = 50
    batch_size = 200

    def __init__(
            self, value, f=64, reg=r'[a-zA-Z0-9_\u4e00-\u9fcc]+', hashfunc=_hashfunc, log=None
    ):
        """
        Initialize a Simhash object.

        Args:
            value: The input to be hashed. Can be a string, another Simhash object,
                  an integer, or an iterable of features.
            f: The dimensions of fingerprints, in bits. Must be a multiple of 8.
            reg: Regular expression for tokenizing strings. Only used when value is a string.
            hashfunc: Function that accepts a utf-8 encoded string and returns either bytes
                     or an integer. If it returns an integer, it should be at least f bits.
            log: Logger instance. If None, a new logger will be created.
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

        Args:
            other: Another Simhash object to compare with

        Returns:
            True if the values are equal, False otherwise
        """
        return self.value == other.value

    def _slide(self, content, width=4):
        """
        Slide a window of the specified width over the content and
        generate n-grams.

        Args:
            content: The string content to slide over
            width: The width of the sliding window

        Returns:
            List of n-grams
        """
        # More robust sliding window that works even with short strings
        if not content:
            return []
        if len(content) <= width:
            return [content]
        else:
            return [content[i:i + width] for i in range(len(content) - width + 1)]

    def _tokenize(self, content):
        """
        Tokenize the content string.

        Args:
            content: The string to tokenize

        Returns:
            List of tokens
        """
        content = content.lower()
        
        # Extract words/characters using the provided regular expression
        words = re.findall(self.reg, content)
        
        # Generate n-grams from the concatenated content
        content = ''.join(words)
        tokens = self._slide(content)
        
        # If the content is too short for n-grams, use the words themselves
        if not tokens and words:
            tokens = words
            
        return tokens

    def build_by_text(self, content):
        """
        Build the simhash value from a string.

        Args:
            content: The text string to build from

        Returns:
            Self, for method chaining
        """
        features = self._tokenize(content)
        features = {k: sum(1 for _ in g) for k, g in groupby(sorted(features))}
        return self.build_by_features(features)

    def build_by_features(self, features):
        """
        Build the simhash value from features.

        Args:
            features: Can be a list of unweighted tokens (weight=1),
                     a list of (token, weight) tuples, or a token -> weight dict.
        
        Returns:
            Self, for method chaining
        """
        # Initialize arrays for summing
        sums = []
        batch = []
        count = 0
        w = 1
        
        # Create a bitmask for ensuring correct-sized values
        mask = (1 << self.f) - 1
        
        # Convert dict format to items 
        if isinstance(features, dict):
            features = features.items()

        # Process each feature
        for f in features:
            skip_batch = True
            if not isinstance(f, basestring):
                f, w = f
                # Large weights or non-integers should bypass batch processing
                skip_batch = w > self.large_weight_cutoff or not isinstance(w, int)

            count += w  # Track total weight
            
            # Get hash value as bytes, handling both integer and bytes returns
            if self.hashfunc_returns_int:
                # Convert integer hash to bytes, limiting size appropriately
                h = int_to_bytes(self.hashfunc(f.encode('utf-8')), self.f_bytes)
            else:
                h = self.hashfunc(f.encode('utf-8'))
                
            # Ensure we get the right number of bytes by taking the last f_bytes
            h = h[-self.f_bytes:]

            # Process features based on their characteristics
            if skip_batch:
                # For large or non-integer weights, process immediately
                if w > 255:
                    # Use float array for precise large weight multiplication
                    bit_array = self._bitarray_from_bytes(h).astype(np.float64) * w
                    sums.append(bit_array)
                else:
                    # For smaller weights, prefer integer operations
                    sums.append(self._bitarray_from_bytes(h) * w)
            else:
                # Add to batch for efficient processing
                batch.extend([h] * w)  # Multiply by weight
                if len(batch) >= self.batch_size:
                    sums.append(self._sum_hashes(batch))
                    batch = []

            # Periodically combine sums to prevent excessive memory usage
            if len(sums) >= self.batch_size:
                sums = [np.sum(sums, axis=0)]

        # Process any remaining batched items
        if batch:
            sums.append(self._sum_hashes(batch))

        # Combine all sums
        if sums:
            combined_sums = np.sum(sums, axis=0)
            
            # Generate a bit array based on whether each bit position's sum 
            # exceeds half the total count (majority vote)
            if count > 0:
                bit_array = combined_sums > count / 2
                
                # Convert bit array to integer, ensuring we only use the lowest f bits
                result_int = bytes_to_int(np.packbits(bit_array).tobytes()) & mask
                self.value = result_int
            else:
                # If no features were processed, default to 0
                self.value = 0
        else:
            # No features to process
            self.value = 0
            
        return self

    def _sum_hashes(self, digests):
        """
        Efficiently sum a batch of hash digests.

        Args:
            digests: List of hash digest byte strings

        Returns:
            Numpy array of summed bits
        """
        if not digests:
            return np.zeros(self.f, dtype=np.int32)
        bitarray = self._bitarray_from_bytes(b''.join(digests))
        # Reshape into a 2D array where each row is a digest's bits
        rows = np.reshape(bitarray, (-1, self.f))
        # Sum each column (bit position)
        return np.sum(rows, axis=0)

    @staticmethod
    def _bitarray_from_bytes(b):
        """
        Convert bytes to a bit array (1s and 0s) using numpy.

        Args:
            b: The bytes to convert

        Returns:
            Numpy array of bits
        """
        if not b:
            return np.array([], dtype=np.uint8)
        return np.unpackbits(np.frombuffer(b, dtype='>B'))

    def distance(self, other):
        """
        Calculate the Hamming distance between two simhashes.
        
        Args:
            other: Another Simhash object to compare with
            
        Returns:
            The hamming distance (number of different bits)
        """
        if not isinstance(other, Simhash):
            raise TypeError("Expected Simhash, got %s" % type(other))
            
        if self.f != other.f:
            raise ValueError(f"Simhash dimensions don't match: {self.f} vs {other.f}")
        
        # XOR the two values to get bits that differ
        x = (self.value ^ other.value) & ((1 << self.f) - 1)
        
        # Use Brian Kernighan's algorithm to count set bits efficiently
        # This is more efficient than naive bit counting for sparse values
        distance = 0
        while x:
            distance += 1
            x &= x - 1  # Clear the least significant bit set
            
        return distance


class SimhashIndex(object):
    """
    Index of Simhash objects that allows for fast lookup of 
    similar hashes through partitioning.
    """
    def __init__(self, objs, f=64, k=2, log=None):
        """
        Initialize the simhash index.

        Args:
            objs: List of (obj_id, simhash) where obj_id is a string, simhash is a Simhash
            f: Number of bits in the simhash (must match the Simhash objects)
            k: Maximum bit difference (hamming distance) to consider as similar
            log: Optional logger instance
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
        Find near-duplicate objects based on simhash distance.

        Args:
            simhash: Simhash object to find near-duplicates for

        Returns:
            List of object IDs that are similar to the input simhash
        """
        if not isinstance(simhash, Simhash):
            raise TypeError("Expected Simhash, got %s" % type(simhash))
            
        if simhash.f != self.f:
            raise ValueError(f"Simhash dimensions don't match: {simhash.f} vs {self.f}")

        ans = set()

        for key in self.get_keys(simhash):
            dups = self.bucket[key]
            self.log.debug('key:%s', key)
            if len(dups) > 200:
                self.log.warning('Big bucket found. key:%s, len:%s', key, len(dups))

            for dup in dups:
                sim2, obj_id = dup.split(',', 1)
                # Convert stored hex string back to Simhash
                sim2 = Simhash(long(sim2, 16), self.f)

                d = simhash.distance(sim2)
                if d <= self.k:
                    ans.add(obj_id)
        return list(ans)

    def add(self, obj_id, simhash):
        """
        Add an object to the index.

        Args:
            obj_id: String identifier for the object
            simhash: Simhash object
        """
        if not isinstance(simhash, Simhash):
            raise TypeError("Expected Simhash, got %s" % type(simhash))
            
        if simhash.f != self.f:
            raise ValueError(f"Simhash dimensions don't match: {simhash.f} vs {self.f}")

        for key in self.get_keys(simhash):
            v = '%x,%s' % (simhash.value, obj_id)
            self.bucket[key].add(v)

    def delete(self, obj_id, simhash):
        """
        Delete an object from the index.

        Args:
            obj_id: String identifier for the object
            simhash: Simhash object
        """
        if not isinstance(simhash, Simhash):
            raise TypeError("Expected Simhash, got %s" % type(simhash))
            
        if simhash.f != self.f:
            raise ValueError(f"Simhash dimensions don't match: {simhash.f} vs {self.f}")

        for key in self.get_keys(simhash):
            v = '%x,%s' % (simhash.value, obj_id)
            if v in self.bucket[key]:
                self.bucket[key].remove(v)

    @property
    def offsets(self):
        """
        Calculate the offsets for the indexing keys.
        Based on the Google paper: http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33026.pdf

        Returns:
            List of offsets
        """
        return [self.f // (self.k + 1) * i for i in range(self.k + 1)]

    def get_keys(self, simhash):
        """
        Generate keys for the index lookup based on the simhash value.
        
        The keys are generated by partitioning the bits of the simhash,
        which ensures similar hashes will collide in at least one bucket.

        Args:
            simhash: A Simhash object

        Returns:
            Iterator of keys for index lookup
        """
        for i, offset in enumerate(self.offsets):
            if i == (len(self.offsets) - 1):
                m = 2 ** (self.f - offset) - 1
            else:
                m = 2 ** (self.offsets[i + 1] - offset) - 1
            c = simhash.value >> offset & m
            yield '%x:%x' % (c, i)

    def bucket_size(self):
        """Get the total number of buckets in the index."""
        return len(self.bucket)
