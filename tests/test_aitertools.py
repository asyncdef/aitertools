"""Test suites for the async itertools features."""

import asyncio
import decimal
import fractions
import functools
import operator

import pytest

import aitertools


def async_test(loop=None):
    """Wrap an async test in a run_until_complete for the event loop."""
    loop = loop or asyncio.get_event_loop()

    def _outer_async_wrapper(func):
        """Closure for capturing the configurable loop."""
        @functools.wraps(func)
        def _inner_async_wrapper(*args, **kwargs):

            return loop.run_until_complete(func(*args, **kwargs))

        return _inner_async_wrapper

    return _outer_async_wrapper


@async_test()
async def test_accumulate_matches_sync():
    """Check if async accumulate matches sync."""
    assert (await aitertools.alist(
        aitertools.accumulate(range(10))
    )) == [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]


@async_test()
async def test_accumulate_kwarg():
    """Check if accumulate works with kwargs."""
    assert (await aitertools.alist(
        aitertools.accumulate(iterable=range(10))
    )) == [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]


@async_test()
@pytest.mark.parametrize(
    'type_',
    (complex, decimal.Decimal, fractions.Fraction)
)
async def test_accumulate_multiple_types(type_):
    """Check if accumulate works with multiple types."""
    assert (await aitertools.alist(
        aitertools.accumulate(map(type_, range(10)))
    )) == list(map(type_, [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]))


@async_test()
async def test_accumulate_non_numeric():
    """Check if accumulate works with non-numeric types."""
    assert (await aitertools.alist(
        aitertools.accumulate('abc')
    )) == ['a', 'ab', 'abc']


@async_test()
async def test_accumulate_empty_iterable():
    """Check if accumulate works with empty iterables."""
    assert (await aitertools.alist(
        aitertools.accumulate([])
    )) == []


@async_test()
async def test_accumulate_single_item():
    """Check if accumulate works on an iterable of one item."""
    assert (await aitertools.alist(
        aitertools.accumulate([7])
    )) == [7]


@async_test()
async def test_accumulate_too_many_args():
    """Check if TypeError is raised on too many args."""
    with pytest.raises(TypeError):

        await aitertools.alist(aitertools.accumulate(range(10), 5, 6))


@async_test()
async def test_accumulate_too_few_args():
    """Check if TypeError is raised on too few args."""
    with pytest.raises(TypeError):

        await aitertools.alist(aitertools.accumulate())


@async_test()
async def test_accumulate_unexpected_kwarg():
    """Check that TypeError is raised when given nonsense kwargs."""
    with pytest.raises(TypeError):

        await aitertools.alist(aitertools.accumulate(x=range(10)))


@async_test()
async def test_accumulate_args_dont_add():
    """Check if TypeError is raised when iterable items don't add."""
    with pytest.raises(TypeError):

        await aitertools.alist(aitertools.accumulate([1, []]))


@async_test()
@pytest.mark.parametrize(
    'seed,func,expected',
    (
        ([2, 8, 9, 5, 7, 0, 3, 4, 1, 6], min, [2, 2, 2, 2, 2, 0, 0, 0, 0, 0]),
        ([2, 8, 9, 5, 7, 0, 3, 4, 1, 6], max, [2, 8, 9, 9, 9, 9, 9, 9, 9, 9]),
        (
            [2, 8, 9, 5, 7, 0, 3, 4, 1, 6],
            operator.mul,
            [2, 16, 144, 720, 5040, 0, 0, 0, 0, 0],
        ),
    ),
)
async def test_accumulate_binary_funcs(seed, func, expected):
    """Check if accumulate correctly applies custom binary functions."""
    assert (await aitertools.alist(
        aitertools.accumulate(seed, func)
    )) == expected


@async_test()
async def test_accumulate_unary_func():
    """Check that TypeError is raised when accumulate with a unary function."""
    with pytest.raises(TypeError):

        await aitertools.alist(aitertools.accumulate(range(10), chr))


@async_test()
async def test_count_starts_at_zero():
    """Check that default counts start as zero."""
    assert (await aitertools.anext(aitertools.count())) == 0


@async_test()
async def test_count_custom_start():
    """Check that count can be give a custom start."""
    assert (await aitertools.anext(aitertools.count(3))) == 3


@async_test()
async def test_count_negative_start():
    """Check that count can start with negative values."""
    assert (await aitertools.anext(aitertools.count(-1))) == -1


@async_test()
async def test_count_steps_one():
    """Check that the default count step is one."""
    counter = aitertools.count()
    assert (await aitertools.anext(counter)) == 0
    assert (await aitertools.anext(counter)) == 1


@async_test()
async def test_count_custom_step():
    """Check that a custom step can be given."""
    counter = aitertools.count(0, 2)
    assert (await aitertools.anext(counter)) == 0
    assert (await aitertools.anext(counter)) == 2


@async_test()
async def test_count_negative_step():
    """Check that count can handle negative step values."""
    counter = aitertools.count(0, -1)
    assert (await aitertools.anext(counter)) == 0
    assert (await aitertools.anext(counter)) == -1


@async_test()
async def test_count_too_many_args():
    """Check that count raises TypeError when too many args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.count(1, 2, 3))


@async_test()
async def test_count_invalid_start():
    """Check that count raises TypeError when start is not a number."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.count('a'))


@async_test()
async def test_cycle_repeats_values():
    """Check if the cycle repeats values on a loop."""
    cycler = aitertools.cycle('abc')
    assert (await aitertools.anext(cycler)) == 'a'
    assert (await aitertools.anext(cycler)) == 'b'
    assert (await aitertools.anext(cycler)) == 'c'
    assert (await aitertools.anext(cycler)) == 'a'
    assert (await aitertools.anext(cycler)) == 'b'
    assert (await aitertools.anext(cycler)) == 'c'


@async_test()
async def test_cycle_is_empty_if_iterable_is_empty():
    """Check if cycle emits no values when given an empty iterable."""
    assert (await aitertools.alist(
        aitertools.cycle('')
    )) == []


@async_test()
async def test_cycle_too_few_args():
    """Check if cycle raises TypeError when too few args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.cycle())


@async_test()
async def test_cycle_non_iterable():
    """Check if cycle raises TypeError when given a non-iterable value."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.cycle(5))


@async_test()
async def test_repeat_repeats_values():
    """Check if a value is emitted multiple times."""
    repeater = aitertools.repeat('a')
    assert (await aitertools.anext(repeater)) == 'a'
    assert (await aitertools.anext(repeater)) == 'a'
    assert (await aitertools.anext(repeater)) == 'a'


@async_test()
async def test_reapeat_can_limit_repititions():
    """Check that repeat limits to a given number."""
    assert (await aitertools.alist(
        aitertools.repeat('a', 3)
    )) == ['a', 'a', 'a']


@async_test()
async def test_repeat_empty_if_zero_times():
    """Check that repeat is empty when given zero times."""
    assert (await aitertools.alist(
        aitertools.repeat('a', 0)
    )) == []


@async_test()
async def test_repeat_empty_if_negative_times():
    """Check that repeat is empty when given negative times."""
    assert (await aitertools.alist(
        aitertools.repeat('a', -1)
    )) == []


@async_test()
async def test_repeat_too_few_args():
    """Check that repeat raises TypeError when too few args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.repeat())


@async_test()
async def test_repeat_too_many_args():
    """Check that repeat raises TypeError when too many args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.repeat('a', 3, 4))


@async_test()
async def test_repeat_invalid_args():
    """Check that repeat raises TypeError when times is non-numeric."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.repeat(None, 'a'))


@async_test()
async def test_chain_combines_iterables():
    """Check that chain produces values from all iterables."""
    assert (await aitertools.alist(
        aitertools.chain('abc', 'def')
    )) == list('abcdef')


@async_test()
async def test_chain_single_iterable():
    """Check if chain works when given only one iterable."""
    assert (await aitertools.alist(
        aitertools.chain('abc')
    )) == list('abc')


@async_test()
async def test_chain_empty_iterable():
    """Check that chain result is empty if all iterables are empty."""
    assert (await aitertools.alist(
        aitertools.chain([])
    )) == []


@async_test()
async def test_chain_non_iterables():
    """Check that chain raises TypeError if given non-iterable values."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.chain(1, 2))


@async_test()
async def test_compress_removes_when_selector_false():
    """Check that compress only emits values with True selectors."""
    assert (await aitertools.alist(
        aitertools.compress('ABCDEF', [1, 0, 1, 0, 1, 1])
    )) == list('ACEF')


@async_test()
async def test_compress_all_true_selectors():
    """Check that compress returns all values when selectors are True."""
    assert (await aitertools.alist(
        aitertools.compress('ABCDEF', range(1, 10))
    )) == list('ABCDEF')


@async_test()
async def test_compress_all_false_selectors():
    """Check that compress is empty when all selectors are False."""
    assert (await aitertools.alist(
        aitertools.compress('ABCDEF', aitertools.repeat(0))
    )) == []


@async_test()
async def test_compress_invalid_args():
    """Check that compress raises TypeError when args aren't iterable."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.compress(None, range(10)))

    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.compress(range(10), None))


@async_test()
async def test_compress_too_few_args():
    """Check that compress raises TypeError when too few args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.compress())


@async_test()
async def test_compress_too_many_args():
    """Check that compress raises TypeError when too many args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.compress([], [], []))


@async_test()
async def test_dropwhile_omits_values_while_predicate_is_true():
    """Check that dropwhile omits values until the predicate returns False."""
    assert (await aitertools.alist(
        aitertools.dropwhile(lambda x: x < 10, range(5, 15))
    )) == list(range(10, 15))


@async_test()
async def test_dropwhile_empty_iterable():
    """Check that dropwhile is empty when the given iterable is empty."""
    assert (await aitertools.alist(
        aitertools.dropwhile(lambda x: x < 10, [])
    )) == []


@async_test()
async def test_dropwhile_too_few_args():
    """Check that dropwhile raises TypeError when too few args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.dropwhile())


@async_test()
async def test_dropwhile_too_many_args():
    """Check that dropwhile raises TypeError when too many args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.dropwhile(1, 2, 3))


@async_test()
async def test_dropwhile_predicate_not_callable():
    """Check that dropwhile raises TypeError if the predicate not callable."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.dropwhile(10, range(10)))


@async_test()
async def test_filterfalse_removes_elements_true_predicate():
    """Check that filterfalse only returns items if the predicate is False."""
    assert (await aitertools.alist(
        aitertools.filterfalse(lambda x: x % 2 == 0, range(6))
    )) == [1, 3, 5]


@async_test()
async def test_filterfalse_defaults_to_bool():
    """Check that filterfalse uses the Truthy/Falsy values by default."""
    assert (await aitertools.alist(
        aitertools.filterfalse(None, [True, False, False, True, False, True])
    )) == [False, False, False]


@async_test()
async def test_filterfalse_too_few_args():
    """Check that filterfalse raises TypeError when too few args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.filterfalse())

    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.filterfalse(None))


@async_test()
async def test_filterfalse_too_many_args():
    """Check that filterfalse raises TypeError when too many args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.filterfalse(1, 2, 3))


@async_test()
async def test_filterfalse_predicate_not_callable():
    """Check that filterfalse raise TypeError when predicate not callable."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.filterfalse([], [1]))


@async_test()
async def test_groupby_empty_iterable():
    """Check that groupby is empty when given an empty iterable."""
    assert (await aitertools.alist(
        aitertools.groupby([])
    )) == []


@async_test()
async def test_groupby_too_few_args():
    """Check that groupby raises TypeError when too few args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.groupby())


@async_test()
async def test_groupby_too_many_args():
    """Check that groupby raises TypeError when too many args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.groupby(1, 2, 3))


@async_test()
async def test_groupby_key_not_callable():
    """Check that groupby raises a TypeError when the key is not callable."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.groupby('abc', []))


@async_test()
async def test_groupby_uses_key_emits_all_values():
    """Check the groupby happy path. Key func is used and all values appear."""
    seed = [
        (0, 10, 20),
        (0, 11, 21),
        (0, 12, 21),
        (1, 13, 21),
        (1, 14, 22),
        (2, 15, 22),
        (3, 16, 23),
        (3, 17, 23)
    ]
    result = []
    async for key, group in aitertools.groupby(seed, lambda r: r[0]):

        async for element in group:

            assert key == element[0]
            result.append(element)

    assert seed == result


@async_test()
async def test_groupby_nested_groupings():
    """Check that groupby works when nesting calls."""
    seed = [
        (0, 10, 20),
        (0, 11, 21),
        (0, 12, 21),
        (1, 13, 21),
        (1, 14, 22),
        (2, 15, 22),
        (3, 16, 23),
        (3, 17, 23)
    ]
    result = []
    async for key1, group1 in aitertools.groupby(seed, lambda r: r[0]):

        async for key2, group2 in aitertools.groupby(group1, lambda r: r[2]):

            async for element in group2:

                assert key1 == element[0]
                assert key2 == element[2]
                result.append(element)

    assert seed == result


@async_test()
async def test_groupby_unused_group_iterable():
    """Check if all keys are found when the group iter is unused."""
    seed = [
        (0, 10, 20),
        (0, 11, 21),
        (0, 12, 21),
        (1, 13, 21),
        (1, 14, 22),
        (2, 15, 22),
        (3, 16, 23),
        (3, 17, 23)
    ]
    result = await aitertools.alist(
        aitertools.groupby(seed, lambda r: r[0])
    )
    result = set(key for key, group in result)
    assert result == set([s[0] for s in seed])


@async_test()
@pytest.mark.parametrize(
    'args',
    (
        (10, 20, 3),
        (10, 3, 20),
        (10, 20),
        (10, 3),
        (20,),
    ),
)
async def test_islice_matches_range(args):
    """Check happy path for islice using equivalent range arguments."""
    assert (await aitertools.alist(
        aitertools.islice(range(100), *args)
    )) == list(range(*args))


@async_test()
@pytest.mark.parametrize(
    'slice_args, range_args',
    (
        ((10, 110, 3), (10, 100, 3)),
        ((10, 110), (10, 100)),
        ((110,), (100,)),
    ),
)
async def test_islice_exausted_iterable(slice_args, range_args):
    """Check that islice stops when the wrapped iterable is exhausted."""
    assert (await aitertools.alist(
        aitertools.islice(range(100), *slice_args)
    )) == list(range(*range_args))


@async_test()
async def test_islice_no_stop():
    """Check that islice exhausts the iterable when no stop is given."""
    assert (await aitertools.alist(
        aitertools.islice(range(10), None)
    )) == list(range(10))

    assert (await aitertools.alist(
        aitertools.islice(range(10), None, None)
    )) == list(range(10))

    assert (await aitertools.alist(
        aitertools.islice(range(10), None, None, None)
    )) == list(range(10))

    assert (await aitertools.alist(
        aitertools.islice(range(10), 2, None)
    )) == list(range(2, 10))

    assert (await aitertools.alist(
        aitertools.islice(range(10), 1, None, 2)
    )) == list(range(1, 10, 2))


@async_test()
async def test_islice_consumption():
    """Check that islice does not overly consume the iterable."""
    iterable = await aitertools.aiter(range(10))
    assert (await aitertools.alist(
        aitertools.islice(iterable, 3)
    )) == list(range(3))
    assert (await aitertools.alist(iterable)) == list(range(3, 10))


@async_test()
@pytest.mark.parametrize(
    'args',
    (
        (-5, 10, 1),
        (1, -5, 1),
        (1, 10, -1),
        (1, 10, 0),
        ('a',),
        ('a', 1),
        (1, 'a'),
        ('a', 1, 1),
        (1, 'a', 1)
    ),
)
async def test_islice_invalid_arguments(args):
    """Check that islice raises ValueError when input is invalid."""
    with pytest.raises(ValueError):

        await aitertools.anext(aitertools.islice(range(10), *args))


@async_test()
async def test_starmap_applies_tuples():
    """Check the starmap happy path."""
    assert (await aitertools.alist(
        aitertools.starmap(operator.pow, ((0, 1), (1, 2), (2, 3)))
    )) == [0**1, 1**2, 2**3]


@async_test()
async def test_starmap_empty_iterable():
    """Check that starmap is empty when the iterable is empty."""
    assert (await aitertools.alist(
        aitertools.starmap(operator.pow, ())
    )) == []


@async_test()
async def test_starmap_non_tuple_iterable_values():
    """Check that starmap handles iterable values that are non-tuple."""
    assert (await aitertools.alist(
        aitertools.starmap(operator.pow, [iter([4, 5])])
    )) == [4**5]


@async_test()
async def test_starmap_too_few_args():
    """Check that starmap raises TypeError when too few args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.starmap())


@async_test()
async def test_starmap_too_many_args():
    """Check that starmap raises TypeError when too many args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.starmap(lambda x: x, range(10), 'a'))


@async_test()
async def test_takewhile_uses_predicate():
    """Check the takewhile happy path. Get values until predicate is False."""
    assert (await aitertools.alist(
        aitertools.takewhile(lambda x: x < 5, range(10))
    )) == [x for x in range(10) if x < 5]


@async_test()
async def test_takewhile_empty_iterable():
    """Check that takewhile is empty when the iterable is empty."""
    assert (await aitertools.alist(
        aitertools.takewhile(lambda x: x < 5, ())
    )) == []


@async_test()
async def test_takewhile_too_few_args():
    """Check that takewhile raises TypeError when too few args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.takewhile())

    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.takewhile(lambda x: x))


@async_test()
async def test_takewhile_too_many_args():
    """Check that takewhile raises TypeError when too many args."""
    with pytest.raises(TypeError):

        await aitertools.anext(aitertools.takewhile(bool, range(1), 'a'))


@async_test()
async def test_tee_empty_iterable():
    """Check that all tees are empty when the iterable is empty."""
    a, b = aitertools.tee([])
    assert (await aitertools.alist(a)) == []
    assert (await aitertools.alist(b)) == []


@async_test()
async def test_tee_when_interleaved():
    """Check that all tees produce correct values when interleaved."""
    a, b = aitertools.tee(range(100))
    a_values, b_values = [], []
    for _ in range(100):

        a_values.append((await aitertools.anext(a)))
        b_values.append((await aitertools.anext(b)))

    assert a_values == b_values
    assert a_values == list(range(100))


@async_test()
async def test_tee_when_not_interleaved():
    """Check that all tees produce correct values when not interleaved."""
    a, b = aitertools.tee(range(10))
    a_values = await aitertools.alist(a)
    b_values = await aitertools.alist(b)

    assert a_values == b_values
    assert a_values == list(range(10))


@async_test()
async def test_tee_can_be_instantiated():
    """Check that new tees can be created from individual tees."""
    a, b = aitertools.tee('abc')
    c = type(a)('def')
    assert (await aitertools.alist(c)) == list('def')


@async_test()
async def test_zip_longest_stops_on_longest_iterable():
    """Check that zip_longest happy path is correct."""
    assert (await aitertools.alist(
        aitertools.zip_longest('ABCD', 'xy', fillvalue='-')
    )) == [('A', 'x'), ('B', 'y'), ('C', '-'), ('D', '-')]


@async_test()
@pytest.mark.parametrize(
    'args,expected',
    (
        ((), [()]),
        (['ab'], [('a',), ('b',)]),
        (
            [range(2), range(3)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        ),
        ([range(0), range(2), range(3)], []),
        ([range(2), range(0), range(3)], []),
        ([range(2), range(3), range(0)], []),
    ),
)
async def test_product_zero_iterables(args, expected):
    """Check that product returns empty tuple when there are no iterables."""
    assert (await aitertools.alist(
        aitertools.product(*args)
    )) == expected


@async_test()
async def test_product_produces_correct_lengths():
    """Check that product produces the right number of results per tuple."""
    assert len((await aitertools.alist(
        aitertools.product(*[range(7)] * 6)
    ))) == 7**6
