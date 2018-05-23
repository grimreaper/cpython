"""Unit tests for the PickleBuffer object.

Pickling tests themselves are in pickletester.py.
"""

import gc
from pickle import PickleBuffer
import sys
import weakref
import unittest

from test import support


class B(bytes):
    pass


class PickleBufferTest(unittest.TestCase):

    def test_constructor_failure(self):
        with self.assertRaises(TypeError):
            PickleBuffer()
        with self.assertRaises(TypeError):
            PickleBuffer("foo")
        # Released memoryview fails taking a buffer
        m = memoryview(b"foo")
        m.release()
        with self.assertRaises(ValueError):
            PickleBuffer(m)

    def test_basics(self):
        pb = PickleBuffer(b"foo")
        self.assertEqual(b"foo", bytes(pb))
        with memoryview(pb) as m:
            self.assertTrue(m.readonly)

        pb = PickleBuffer(bytearray(b"foo"))
        self.assertEqual(b"foo", bytes(pb))
        with memoryview(pb) as m:
            self.assertFalse(m.readonly)
            m[0] = 48
        self.assertEqual(b"0oo", bytes(pb))

    def test_release(self):
        pb = PickleBuffer(b"foo")
        pb.release()
        with self.assertRaises(ValueError) as raises:
            memoryview(pb)
        self.assertIn("operation forbidden on released PickleBuffer object",
                      str(raises.exception))
        # Idempotency
        pb.release()

    def test_cycle(self):
        b = B(b"foo")
        pb = PickleBuffer(b)
        b.cycle = pb
        wpb = weakref.ref(pb)
        del b, pb
        gc.collect()
        self.assertIsNone(wpb())

    def test_ndarray(self):
        ndarray = support.import_module("_testbuffer").ndarray
        arr = ndarray(list(range(12)), shape=(4, 3), format='<i')
        pb = PickleBuffer(arr[::2])
        with memoryview(pb) as m:
            self.assertTrue(m.readonly)
            self.assertEqual(m.itemsize, 4)
            self.assertEqual(m.shape, (2, 3))
            self.assertEqual(m.strides, (3 * 2 * m.itemsize, 4))
            self.assertFalse(m.c_contiguous)
            self.assertFalse(m.f_contiguous)


if __name__ == "__main__":
    unittest.main()
