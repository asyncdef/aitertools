==========
aitertools
==========

*Async versions of the itertools features.*

Example Usage
=============

The behaviour of each feature in this project is designed to match, one-to-one,
the behaviour of each feature in the Python `itertools` module. The primary
difference of this package is that all features can additionally consume
async functions and async iterables in addition to the sync versions. All
functions also return async iterables.

Wrapping Sync Code
------------------

When working with sync iterables in an async world you may want to wrap things
in an async interface for interoperability with other async tools. For example,
a typical sync call to `chain` might look like:

.. code-block:: python

    iter1 = (1, 2, 3, 4)
    iter2 = (5, 6, 7, 8)
    iter3 = (9, 10, 11, 12)

    for value in itertools.chain(iter1, iter2, iter3):

        print(value)

The above would output numbers 1 - 12 in the order shown. However, in the async
world not all iterables are tuples and lists. The same thing can be
accomplished with the async `chain`:

.. code-block:: python

    iter1 = (1, 2, 3, 4)
    iter2 = (5, 6, 7, 8)
    iter3 = (9, 10, 11, 12)

    async for value in aitertools.chain(iter1, iter2, iter3):

        print(value)

The above behaves exactly as the sync version except that it exposes the async
iteratable interface and works with `async def`.

Mixing Sync and Async
---------------------

All functions in the `aitertools` package accept both sync and async iterables.
This allows for the combination of the two when needed:

.. code-block:: python

    iter1 = (1, 2, 3, 4)
    iter2 = some_async_iter()  # Async resolves to 5, 6, 7, 8

    async for value in aitertools.chain(iter1, iter2):

        print(value)

Ensuring Async Interfaces
-------------------------

This package also provides a handful of tools for working with the async
interfaces:

    -   aiter

        Async function that counters `iter`. This function, when given an
        async iterable, will return an async iterator. When given a sync
        iterable it will wrap it and return an async iterator.

    -   anext

        Async function that counters `next`. This function calls the
        `__anext__()` method of the iterator and returns the value. It,
        like the `next` method, can optionally return a default value when
        the iterator is empty.

    -   alist

        Async function that counters `list`. For use cases like
        `list(iterable)` this function allows for `await alist(iterable)`.

    -   atuple

        Async function that counters `tuple`. Similar to the `alist` above.

Development Roadmap
===================

The current release of this package is only missing the `permutations`,
`combinations`, and `combinations_with_replacement` features. The next release
should contain these missing features.

Additionally, several features from functools and the available globals are
slated for addition. For example, the `filter`, `map`, and `reduce` features
are good fits for this package and already have at least one `itertools`
counterpart added.

Testing
=======

All tests are stored in the '/tests' subdirectory. All tests are expected to
pass for Python 3.5 and above. To run tests create a virtualenv and install
the test-requirements.txt list. After that using the `tox` command will launch
the test suite.

License
=======

    Copyright 2015 Kevin Conway

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Contributing
============

Firstly, if you're putting in a patch then thank you! Here are some tips for
getting your patch merged:

Style
-----

As long as the code passes the PEP8 and PyFlakes gates then the style is
acceptable.

Docs
----

The PEP257 gate will check that all public methods have docstrings. If you're
adding additional features from the `itertools` module try to preserve the
original docstrings if possible. Sometimes the original docs don't fit well
with the PEP257 format. In those cases it is OK to modify the docstring to fit.
If you're adding something new, like a helper function, try out the
`napoleon style of docstrings <https://pypi.python.org/pypi/sphinxcontrib-napoleon>`_.

Tests
-----

Make sure the patch passes all the tests. If you're adding a new feature from
`itertools` be sure to add some of the original, standard library tests. The
tests used to validate the Python `itertools` module are found in the
'/Lib/test/test_itertools.py' file in the Python source. The orginal tests are
organized into large blocks of tests based on features. As much as possible,
break the tests up into individual units. Check out the existing tests for this
project for inspiration if needed.

If you're adding something totally new the make sure to throw in a few tests.
If you're fixing a bug then definitely add at least one test to prevent
regressions.
