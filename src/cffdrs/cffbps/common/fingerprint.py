"""Exact-input fingerprints used by the foliar moisture result cache."""
import hashlib

import numpy as np


def fingerprint(*values) -> bytes:
    """Return a 16-byte BLAKE2b digest of arrays and scalars, by exact content.

    Each array contributes its shape, dtype, data bytes and (for masked arrays) mask
    bytes, so a change of value, mask, dtype or shape changes the digest. ``None`` and
    plain scalars hash by value. No copy of an array is kept.
    """
    h = hashlib.blake2b(digest_size=16)
    for value in values:
        if value is None:
            h.update(b'N')
        elif isinstance(value, np.ndarray):
            data = np.ma.getdata(value)
            h.update(b'A')
            h.update(repr((data.shape, data.dtype.str)).encode())
            h.update(memoryview(np.ascontiguousarray(data).reshape(-1)).cast('B'))
            if isinstance(value, np.ma.MaskedArray) and value.mask is not np.ma.nomask:
                h.update(b'M')
                h.update(memoryview(np.ascontiguousarray(np.ma.getmaskarray(value)).reshape(-1)).cast('B'))
            else:
                h.update(b'm')
        elif isinstance(value, (bool, int, float, np.integer, np.floating, str)):
            h.update(b'S')
            h.update(repr((type(value).__name__, value)).encode())
        else:
            raise TypeError(f'cannot fingerprint a {type(value).__name__}')
    return h.digest()
