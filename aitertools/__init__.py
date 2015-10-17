"""Async versions of the itertools features."""

import collections
import functools
import itertools as sync_itertools
import operator
import types


class AsyncIterWrapper:

    """Async wrapper for synchronous iterables."""

    def __init__(self, iterable):
        """Initialize the wrapper with some iterable."""
        self._iterable = iter(iterable)

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Fetch the next value from the iterable."""
        try:

            return next(self._iterable)

        except StopIteration as exc:

            raise StopAsyncIteration() from exc

    def __repr__(self):
        """Get a human representation of the wrapper."""
        return '<AsyncIterWrapper {}>'.format(self._iterable)


async def aiter(*args):
    """Return an iterator object.

    Args:
        obj: An object that implements the __iter__ or __aiter__ method.
        sentinel: An optional sentinel value to look for while iterator.

    Return:
        iterable: Some iterable that provides a __anext__ method.

    Raises:
        TypeError: If only the object is given and it is not iterable.
        TypeError: If two arguments are given and the first is not an async
            callable.

    This function behaves very differently based on the number of arguments
    given. If only the first argument is present the method will return
    an async iterable that implements the __anext__ method by called the
    given object's __aiter__. If the object does not define __aiter__ but does
    define __iter__ then the result will be an AsyncIterWrapper that contains
    the original iterable. This form of the function can be used to coerce all
    iterables, async or not, into async iterables for interoperablilty.

    If the second argument is given then the first argument _must_ be an async
    callable. The returned value will still be an iterable implementing the
    __aiter__ method, but each call to that method will call the underlying
    async callable. If the value returned from the async callable matches the
    sentinel value then StopAsyncIteration is raised. Otherwise the value is
    returned.
    """
    if not args:

        raise TypeError('aiter() expected at least 1 arguments, got 0')

    if len(args) > 2:

        raise TypeError(
            'aiter() expected at most 2 arguments, got {}'.format(len(args))
        )

    if len(args) == 2:

        func, sentinel = args
        if not isinstance(func, types.CoroutineType):

            raise TypeError('aiter(v, w): v must be async callable')
        # TODO: repeating call thing
        raise NotImplementedError()

    obj = args[0]
    if hasattr(obj, '__anext__'):

        return obj

    if hasattr(obj, '__aiter__'):

        return (await obj.__aiter__())

    if hasattr(obj, '__iter__') or hasattr(obj, '__next__'):

        return AsyncIterWrapper(iter(obj))

    raise TypeError("'{}' object is not iterable".format(type(args[0])))


async def anext(*args):
    """Return the next item from an async iterator.

    Args:
        iterable: An async iterable.
        default: An optional default value to return if the iterable is empty.

    Return:
        The next value of the iterable.

    Raises:
        TypeError: The iterable given is not async.

    This function will return the next value form an async iterable. If the
    iterable is empty the StopAsyncIteration will be propogated. However, if
    a default value is given as a second argument the exception is silenced and
    the default value is returned instead.
    """
    if not args:

        raise TypeError('anext() expected at least 1 arguments, got 0')

    if len(args) > 2:

        raise TypeError(
            'anext() expected at most 2 arguments, got {}'.format(len(args))
        )

    iterable, default, has_default = args[0], None, False
    if len(args) == 2:

        iterable, default = args
        has_default = True

    try:

        return await iterable.__anext__()

    except StopAsyncIteration as exc:

        if has_default:

            return default

        raise StopAsyncIteration() from exc


async def alist(iterable):
    """Async standin for the list built-in.

    This function consumes an async iterable and returns a list of values
    resolved from the iterable.
    """
    values = []
    async for value in iterable:

        values.append(value)

    return values


async def atuple(iterable):
    """Async standin for the tuple built-in.

    This function consumes an async iterable and returns a tuple of values
    resolved from the iterable.
    """
    return tuple((await alist(iterable)))


def count(start=0, step=1):
    """Make an iterator that returns evenly spaced values."""
    return AsyncIterWrapper(sync_itertools.count(start, step))


class AsyncCycle:

    """Async version of the cycle iterable."""

    def __init__(self, iterable):
        """Initialize the cycle with some iterable."""
        self._values = []
        self._iterable = iterable
        self._initialized = False
        self._depleted = False
        self._offset = 0

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next value of the iterable or one from cache."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True

        if self._depleted:

            offset, self._offset = self._offset, self._offset + 1
            if self._offset >= len(self._values):

                self._offset = 0

            return self._values[offset]

        try:

            value = await anext(self._iterable)
            self._values.append(value)
            return value

        except StopAsyncIteration as exc:

            self._depleted = True
            if not self._values:

                raise StopAsyncIteration() from exc

            self._offset = 1
            return self._values[0]

    def __repr__(self):
        """Get a human representation of the cycle."""
        return '<AsyncCycle {}>'.format(self._iterable)


def cycle(iterable):
    """Repeat all elements of the iterable forever.

    Make an iterator returning elements from the iterable and saving a copy of
    each. When the iterable is exhausted, return elements from the saved copy.
    Repeats indefinitely.
    """
    return AsyncCycle(iterable)


def repeat(obj, times=None):
    """Make an iterator that returns object over and over again."""
    if times is None:

        return AsyncIterWrapper(sync_itertools.repeat(obj))

    return AsyncIterWrapper(sync_itertools.repeat(obj, times))


def _async_callable(func):
    """Ensure the callable is an async def."""
    if isinstance(func, types.CoroutineType):

        return func

    @functools.wraps(func)
    async def _async_def_wrapper(*args, **kwargs):
        """Wrap a a sync callable in an async def."""
        return func(*args, **kwargs)

    return _async_def_wrapper


class AsyncAccumulate:

    """Async verion of the accumulate iterable."""

    def __init__(self, iterable, func=operator.add):
        """Initialize the wrapper with an iterable and binary function."""
        self._iterable = iterable
        self._func = _async_callable(func)
        self._initialized = False
        self._started = False
        self._total = None
        self._depleted = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next accumulated value."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True

        if not self._started:

            self._started = True
            try:

                self._total = await anext(self._iterable)

            except StopAsyncIteration as exc:

                self._depleted = True
                raise StopAsyncIteration() from exc

            return self._total

        if self._depleted:

            raise StopAsyncIteration()

        try:

            next_value = await anext(self._iterable)

        except StopAsyncIteration as exc:

            self._depleted = True
            raise StopAsyncIteration() from exc

        self._total = await self._func(self._total, next_value)
        return self._total


def accumulate(iterable, func=operator.add):
    """Make an iterable that returns accumulated sums.

    An optional second argument can be given to run a custom binary function.
    If func is supplied, it should be a function of two arguments. Elements of
    the input iterable may be any type that can be accepted as arguments to
    func. (For example, with the default operation of addition, elements may be
    any addable type including Decimal or Fraction.) If the input iterable is
    empty, the output iterable will also be empty.
    """
    return AsyncAccumulate(iterable, func)


class AsyncChain:

    """Async version of the chain iterable."""

    def __init__(self, *iterables):
        """Initialize the wrapper with some number of iterables."""
        self._iterables = iterables
        self._current = None
        self._initialized = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next value in the chain."""
        if not self._initialized:

            current_iterables, self._iterables = self._iterables, []
            for iterable in current_iterables:

                self._iterables.append((await aiter(iterable)))

            self._iterables = collections.deque(self._iterables)
            self._initialized = True
            self._current = self._iterables.popleft()

        while True:

            if not self._iterables and not self._current:

                raise StopAsyncIteration()

            try:

                return await anext(self._current)

            except StopAsyncIteration:

                self._current = None
                if self._iterables:

                    self._current = self._iterables.popleft()


def chain(*iterables):
    """Combine iteratators into one stream of values.

    Make an iterator that returns elements from the first iterable until it is
    exhausted, then proceeds to the next iterable, until all of the iterables
    are exhausted. Used for treating consecutive sequences as a single
    sequence.
    """
    return AsyncChain(*iterables)


def chain_from_iterable(iterable):
    """Chain iterables contained within a lazily evaluated iterable."""
    raise NotImplementedError()


chain.from_iterable = chain_from_iterable


class AsyncCompress:

    """Async version of the compress iterable."""

    def __init__(self, data, selectors):
        """Initialize the iterable with data and selectors."""
        self._data = data
        self._selectors = selectors
        self._initialized = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Fetch the next selected value."""
        if not self._initialized:

            self._data = await aiter(self._data)
            self._selectors = await aiter(self._selectors)
            self._initialized = True

        while True:

            try:

                value = await anext(self._data)
                selection = await anext(self._selectors)

                if selection:

                    return value

            except StopAsyncIteration as exc:

                raise StopAsyncIteration() from exc


def compress(data, selectors):
    """Only return elements from data with a corresponding True selector.

    Make an iterator that filters elements from data returning only those that
    have a corresponding element in selectors that evaluates to True. Stops
    when either the data or selectors iterables has been exhausted.
    """
    return AsyncCompress(data, selectors)


class AsyncDropWhile:

    """Async version of the dropwhile iterable."""

    def __init__(self, predicate, iterable):
        """Initialize the iterable with a predicate and data iterable."""
        self._predicate = _async_callable(predicate)
        self._iterable = iterable
        self._initialized = False
        self._found = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get a value after the test returns False."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True

        while not self._found:

            value = await anext(self._iterable)
            self._found = not (await self._predicate(value))
            if self._found:

                return value

        return await anext(self._iterable)


def dropwhile(predicate, iterable):
    """Skip values from iterable until predicate returns False.

    Make an iterator that drops elements from the iterable as long as the
    predicate is true; afterwards, returns every element. Note, the iterator
    does not produce any output until the predicate first becomes false, so it
    may have a lengthy start-up time.
    """
    return AsyncDropWhile(predicate, iterable)


class AsyncFilterFalse:

    """Async version of the filterfalse iterable."""

    def __init__(self, predicate, iterable):
        """Initialize the iterable with a predicate and data iterable."""
        self._predicate = predicate
        if self._predicate is None:

            self._predicate = bool

        self._predicate = _async_callable(self._predicate)
        self._iterable = iterable
        self._initialized = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next value for which predicate returns True."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True

        while True:

            value = await anext(self._iterable)
            test = await self._predicate(value)
            if not test:

                return value


def filterfalse(predicate, iterable):
    """Only emit values for which the predicate returns Flase.

    Make an iterator that filters elements from iterable returning only those
    for which the predicate is False. If predicate is None, return the items
    that are false.
    """
    return AsyncFilterFalse(predicate, iterable)


class _AsyncGroupByGroupIterable:

    """Async version of the group from groupby."""

    def __init__(self, group_by):
        self._group_by = group_by
        self._initialized = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next value in the group."""
        if not self._initialized:

            self._initialized = True
            return self._group_by._current_value

        try:

            value = await anext(self._group_by._iterable)

        except StopAsyncIteration as exc:

            self._group_by._stop = True
            raise StopAsyncIteration() from exc

        key = await self._group_by._key(value)
        if key == self._group_by._current_key:

            return value

        self._group_by._current_value = value
        self._group_by._current_key = key
        raise StopAsyncIteration()


class AsyncGroupBy:

    """Async version of the groupby iterable."""

    def __init__(self, iterable, key=None):
        """Initialize the iterable with a data and optional key function."""
        self._iterable = iterable
        self._key = key
        if self._key is None:

            self._key = lambda x: x

        self._key = _async_callable(self._key)
        self._initialized = False
        self._singleton = object()
        self._current_key = self._singleton
        self._current_value = self._singleton
        self._stop = False
        self._group_iter = None

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next group in the iterable."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True

        if self._stop:

            raise StopAsyncIteration()

        if self._current_value is self._singleton:

            self._current_value = await anext(self._iterable)
            self._current_key = await self._key(self._current_value)

        if self._group_iter and not self._group_iter._initialized:

            value = await anext(self._iterable)
            key = await self._key(value)
            while key == self._current_key:

                value = await anext(self._iterable)
                key = await self._key(value)

            self._current_value = value
            self._current_key = key

        self._group_iter = _AsyncGroupByGroupIterable(self)
        return (self._current_key, self._group_iter)


def groupby(iterable, key=None):
    """Make an iterator that returns consecutive keys and groups.

    The key is a function computing a key value for each element. If not
    specified or is None, key defaults to an identity function and returns the
    element unchanged. Generally, the iterable needs to already be sorted on
    the same key function.

    The operation of groupby() is similar to the uniq filter in Unix. It
    generates a break or new group every time the value of the key function
    changes (which is why it is usually necessary to have sorted the data using
    the same key function). That behavior differs from SQL’s GROUP BY which
    aggregates common elements regardless of their input order.

    The returned group is itself an iterator that shares the underlying
    iterable with groupby(). Because the source is shared, when the groupby()
    object is advanced, the previous group is no longer visible. So, if that
    data is needed later, it should be stored as a list.
    """
    return AsyncGroupBy(iterable, key)


class AsyncISlice:

    """Async version of the islice iterable."""

    def __init__(self, iterable, *args):
        """Inititalize the iterable with start, stop, and step values."""
        if not args:

            raise TypeError('islice expected at least 2 arguments, got 1')

        if len(args) == 1:

            start, stop, step = 0, args[0], 1

        if len(args) == 2:

            start, stop = args
            step = 1

        if len(args) == 3:

            start, stop, step = args

        if len(args) > 3:

            raise TypeError(
                'islice expected at most 4 arguments, got {}'.format(len(args))
            )

        start = start if start is not None else 0
        step = step if step is not None else 1

        if start is not None and not isinstance(start, int):

            raise ValueError('The start value must be an integer.')

        if stop is not None and not isinstance(stop, int):

            raise ValueError('The stop value must be an integer.')

        if step is not None and not isinstance(step, int):

            raise ValueError('The step value must be an integer.')

        if start < 0:

            raise ValueError('The start value cannot be negative.')

        if stop is not None and stop < 0:

            raise ValueError('The stop value cannot be negative.')

        if step < 1:

            raise ValueError('The step value cannot be negative or zero.')

        self._start, self._step, self._stop = start, step, stop
        self._iterable = iterable
        self._initialized = False
        self._offset = 0
        self._depleted = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next value in the slice."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True
            for _ in range(self._start):

                await anext(self._iterable)
                self._offset = self._offset + 1

            value = await anext(self._iterable)
            self._offset = self._offset + 1
            if self._stop is not None and self._offset >= self._stop:

                raise StopAsyncIteration()

            return value

        for _ in range(self._step - 1):

            await anext(self._iterable)
            self._offset = self._offset + 1

        if self._stop is not None and self._offset >= self._stop:

            raise StopAsyncIteration()

        value = await anext(self._iterable)
        self._offset = self._offset + 1
        return value


def islice(iterable, *args):
    """Make an iterator that returns selected elements from the iterable.

    If start is non-zero, then elements from the iterable are skipped until
    start is reached. Afterward, elements are returned consecutively unless
    step is set higher than one which results in items being skipped. If stop
    is None, then iteration continues until the iterator is exhausted, if at
    all; otherwise, it stops at the specified position. Unlike regular slicing,
    islice() does not support negative values for start, stop, or step. Can be
    used to extract related fields from data where the internal structure has
    been flattened (for example, a multi-line report may list a name field on
    every third line).
    """
    return AsyncISlice(iterable, *args)


class AsyncStarMap:

    """Async version of the starmap iterable."""

    def __init__(self, func, iterable):
        """Initialize the iterable with a func and data."""
        self._func = _async_callable(func)
        self._iterable = iterable
        self._initialized = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next mapped value."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True

        args = await anext(self._iterable)
        return await self._func(*args)


def starmap(func, iterable):
    """Make an iterator that calls func using arguments from the iterable.

    Used instead of map() when argument parameters are already grouped in
    tuples from a single iterable (the data has been “pre-zipped”). The
    difference between map() and starmap() parallels the distinction between
    function(a,b) and function(*c).
    """
    return AsyncStarMap(func, iterable)


class AsyncTakeWhile:

    """Async version of the takewhile iterable."""

    def __init__(self, predicate, iterable):
        """Initialize the iterable with a predicate and data iterable."""
        self._predicate = _async_callable(predicate)
        self._iterable = iterable
        self._initialized = False
        self._stop = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get a value after the test returns True."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True

        if self._stop:

            raise StopAsyncIteration()

        value = await anext(self._iterable)
        self._stop = not (await self._predicate(value))

        if self._stop:

            raise StopAsyncIteration()

        return value


def takewhile(predicate, iterable):
    """Return values while the predicate returns true.

    Make an iterator that returns elements from the iterable as long as the
    predicate is true.
    """
    return AsyncTakeWhile(predicate, iterable)


class AsyncTeeIterable:

    """Async version of the tee iterable."""

    def __init__(self, iterable):
        """Initialize the iterable with data and a number of tees."""
        self._iterable = iterable
        self._siblings = ()
        self._initialized = False
        self._cache = collections.deque()

    def _append(self, value):
        """Add a value to the internal cache."""
        self._cache.append(value)

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next value in the tee."""
        if not self._initialized:

            self._iterable = await aiter(self._iterable)
            self._initialized = True
            for sibling in self._siblings:

                sibling._iterable = self._iterable
                sibling._initialized = True

        if self._cache:

            return self._cache.popleft()

        value = await anext(self._iterable)
        for sibling in self._siblings:

            if sibling is self:

                continue

            sibling._append(value)

        return value


def tee(iterable, n=2):
    """Return n independent iterators from a single iterable.

    Once tee() has made a split, the original iterable should not be used
    anywhere else; otherwise, the iterable could get advanced without the tee
    objects being informed.

    This itertool may require significant auxiliary storage (depending on how
    much temporary data needs to be stored). In general, if one iterator uses
    most or all of the data before another iterator starts, it is faster to use
    list() instead of tee().
    """
    tees = tuple(AsyncTeeIterable(iterable) for _ in range(n))
    for tee in tees:

        tee._siblings = tees

    return tees


class _ZipExhausted(Exception):

    """Internal exception for signaling zip complete."""


class AsyncZipLongest:

    """Async version of the zip_longest iterable."""

    def __init__(self, *iterables, fillvalue=None):
        """Initialize with content to zip and a fill value."""
        self._iterables = iterables
        self._fillvalue = fillvalue
        self._initialized = False
        self._remaining = len(self._iterables)

    async def __aiter__(self):
        """Return self."""
        return self

    def _iterable_exhausted(self):
        self._remaining = self._remaining - 1
        if not self._remaining:

            raise _ZipExhausted()

        yield self._fillvalue

    async def __anext__(self):
        """Get the next zip of values."""
        if not self._initialized:

            fillers = repeat(self._fillvalue)
            chained_iters = []
            for iterable in self._iterables:

                chained_iters.append(
                    (await aiter(
                        chain(
                            iterable,
                            self._iterable_exhausted(),
                            fillers
                        )
                    ))
                )

            self._iterables = chained_iters

        if not self._remaining:

            raise StopAsyncIteration()

        values = []
        try:

            for iterable in self._iterables:

                values.append((await anext(iterable)))

        except _ZipExhausted:

            raise StopAsyncIteration()

        return tuple(values)


def zip_longest(*iterables, fillvalue=None):
    """Make an iterator that aggregates elements from each of the iterables.

    If the iterables are of uneven length, missing values are filled-in with
    fillvalue. Iteration continues until the longest iterable is exhausted.
    """
    return AsyncZipLongest(*iterables, fillvalue=fillvalue)


class AsyncProduct:

    """Async version of the product iterable."""

    def __init__(self, *iterables, repeat=1):
        """Initialize with data and a repeat value."""
        self._iterables = iterables
        self._offsets = []
        self._repeat = repeat
        self._initialized = False
        self._stop = False

    async def __aiter__(self):
        """Return self."""
        return self

    async def __anext__(self):
        """Get the next product in the iterable."""
        if not self._initialized:

            if not self._iterables:

                self._initialized = True
                self._stop = True
                return ()

            iterables = []
            for iterable in self._iterables:

                iterable = await aiter(iterable)
                iterable = await alist(iterable)
                if not iterable:

                    self._initialized = True
                    self._stop = True
                    raise StopAsyncIteration()

                iterables.append(iterable)

            self._iterables = iterables * self._repeat
            self._offsets = [0 for _ in self._iterables]
            self._initialized = True
            return tuple(iterable[0] for iterable in self._iterables)

        if self._stop:

            raise StopAsyncIteration()

        for offset, iterable in enumerate(reversed(self._iterables), start=1):

            self._offsets[-offset] = self._offsets[-offset] + 1
            if self._offsets[-offset] >= len(iterable):

                self._offsets[-offset] = 0
                if offset >= len(self._iterables):

                    self._stop = True
                    raise StopAsyncIteration()

                continue

            return tuple(
                iterable[offset]
                for iterable, offset in zip(self._iterables, self._offsets)
            )


def product(*iterables, repeat=1):
    """Cartesian product of input iterables.

    Equivalent to nested for-loops in a generator expression. For example,
    product(A, B) returns the same as ((x,y) for x in A for y in B).

    The nested loops cycle like an odometer with the rightmost element
    advancing on every iteration. This pattern creates a lexicographic
    ordering so that if the input’s iterables are sorted, the product tuples
    are emitted in sorted order.

    To compute the product of an iterable with itself, specify the number of
    repetitions with the optional repeat keyword argument. For example,
    product(A, repeat=4) means the same as product(A, A, A, A).
    """
    return AsyncProduct(*iterables, repeat=repeat)
