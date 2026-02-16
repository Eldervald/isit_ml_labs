from dataclasses import is_dataclass, FrozenInstanceError, asdict
from typing import Any
from pathlib import Path
import re

import pytest

from .orders import Item, Position, CountedPosition, WeightedPosition, Order


@pytest.mark.parametrize('class_type', [
    Item, Position, CountedPosition, WeightedPosition, Order
])
def test_class_type(class_type: Any) -> None:
    assert is_dataclass(class_type)


def test_no_init_implemented() -> None:
    solution_file = Path(__file__).parent / 'orders.py'
    assert solution_file.exists()

    with open(solution_file) as f:
        for i in f:
            assert not re.match('.*__init__.*', i), 'You should not use __init__ in dataclasses'


def test_no_compare_implemented() -> None:
    solution_file = Path(__file__).parent / 'orders.py'
    assert solution_file.exists()

    with open(solution_file) as f:
        for i in f:
            for method in ['__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__']:
                assert not re.match(f'.*{method}.*', i), f'You should not use {method} in this task. Do it with field()'


def test_item_params_check() -> None:
    Item(item_id=-1, title='Spoon', cost=25)

    with pytest.raises(AssertionError):
        Item(item_id=100, title='', cost=25)

    with pytest.raises(AssertionError):
        Item(item_id=10, title='Pen', cost=-25)
    with pytest.raises(AssertionError):
        Item(item_id=10, title='Another Pen', cost=0)


def test_item_frozen() -> None:
    item = Item(item_id=0, cost=500, title='Sub-Zero')
    with pytest.raises(FrozenInstanceError):
        item.item_id = 10  # type: ignore


def test_items_ordering() -> None:
    assert Item(item_id=0, title='Pop-it', cost=200) < Item(item_id=1, title='Simple Dimple', cost=200)
    assert Item(item_id=0, title='Jacket', cost=15000) > Item(item_id=1, title='Jacket', cost=400)


def test_items_sort() -> None:
    items = [
        Item(item_id=9, title='Thing', cost=44),
        Item(item_id=0, title='Note', cost=8),
        Item(item_id=1, title='Things', cost=100),
        Item(item_id=8, title='Unity', cost=5),
        Item(item_id=11, title='Things', cost=44),
        Item(item_id=15, title='Note', cost=64),
    ]
    assert [i.item_id for i in sorted(items)] == [0, 15, 9, 11, 1, 8]


@pytest.mark.parametrize('class_type', [
    CountedPosition, WeightedPosition
])
def test_position_inheritance(class_type: Any) -> None:
    assert issubclass(class_type, Position)


def test_position_is_abstract() -> None:
    item = Item(item_id=0, title='Spoon', cost=25)

    assert getattr(Position.cost, '__isabstractmethod__', False), '`cost` should be an abstractmethod'

    with pytest.raises(TypeError) as e:
        _ = Position(item=item)  # type: ignore
    assert (
        "Can't instantiate abstract class Position without an implementation for abstract method 'cost'"
        in str(e.value)
    )


@pytest.mark.parametrize('class_, input_, expected_cost', [
    (CountedPosition, dict(item=Item(0, 'USB cable', 256)), 256),
    (CountedPosition, dict(item=Item(0, 'USB cable', 256), count=4), 1024),
    (CountedPosition, dict(item=Item(0, 'USB plug', 256), count=2), 512),
    (CountedPosition, dict(item=Item(0, 'Book', 4), count=20), 80),
    (WeightedPosition, dict(item=Item(0, 'Book', 40)), 40),
    (WeightedPosition, dict(item=Item(0, 'Book', 4), weight=20), 80),
    (WeightedPosition, dict(item=Item(0, 'Shugar', 256), weight=0.5), 128),
    (WeightedPosition, dict(item=Item(0, 'Melon', 40), weight=8.3), 332),
])
def test_position_cost(class_: type, input_: dict[str, Any], expected_cost: int) -> None:
    position = class_(**input_)
    assert 'cost' not in asdict(position), '`cost` should be a property'
    assert position.cost == expected_cost
    assert isinstance(position.cost, float) or isinstance(position.cost, int)


@pytest.mark.parametrize('input_, expected_cost', [
    (dict(positions=[CountedPosition(Item(0, 'USB cable', 256), count=4)]), 1024),
    (dict(positions=[CountedPosition(Item(0, 'USB cable', 256), count=2)], have_promo=True), 435),
    (dict(positions=[CountedPosition(Item(i, 'Book', i * 100), count=i) for i in range(5, 8)]), 11000),
    (dict(positions=[CountedPosition(Item(i, 'Book', i * 50)) for i in range(1, 3)], have_promo=True), 127),
    (dict(positions=[WeightedPosition(Item(0, 'Melon', 40), weight=8.3)]), 332),
    (dict(positions=[WeightedPosition(Item(0, 'Melon', 40), weight=8.3), CountedPosition(Item(0, 'Box', 90))]), 422),
])
def test_order_cost(input_: dict[str, Any], expected_cost: int) -> None:
    input_['order_id'] = 0
    order = Order(**input_)
    assert order.cost == expected_cost
    assert isinstance(order.cost, int)


def test_order_have_promo_is_not_field() -> None:
    order = Order(order_id=0, have_promo=True)
    assert 'have_promo' not in asdict(order)


def test_order_no_positions() -> None:
    order_first = Order(order_id=0)
    order_first.positions.append(CountedPosition(Item(0, 'USB cable', 256)))
    order_second = Order(order_id=1)
    assert order_first.positions != order_second.positions


###################
# Additional edge case tests
###################


def test_position_zero_count_weight() -> None:
    # Edge case: zero count/weight for positions
    item = Item(item_id=0, title='Test Item', cost=100)

    # Zero count should result in zero cost
    pos = CountedPosition(item=item, count=0)
    assert pos.cost == 0

    # Zero weight should result in zero cost
    pos2 = WeightedPosition(item=item, weight=0)
    assert pos2.cost == 0


def test_order_large_many_positions() -> None:
    # Edge case: very large orders with many positions
    positions = [CountedPosition(Item(i, f'Item{i}', 10), count=i+1) for i in range(100)]
    order = Order(order_id=1, positions=positions)

    # Verify total cost is calculated correctly
    expected_cost = sum(pos.cost for pos in positions)
    assert order.cost == expected_cost


def test_order_mixed_position_types() -> None:
    # Edge case: mixed position types in single order
    item1 = Item(item_id=0, title='Cable', cost=100)
    item2 = Item(item_id=1, title='Book', cost=50)

    positions = [
        CountedPosition(item=item1, count=2),  # 200
        WeightedPosition(item=item2, weight=1.5),  # 75
        CountedPosition(item=item2, count=1),  # 50
    ]

    order = Order(order_id=2, positions=positions)
    assert order.cost == 325


def test_item_unicode_titles() -> None:
    # Edge case: Unicode titles for items
    items = [
        Item(item_id=0, title='Товар', cost=100),  # Russian
        Item(item_id=1, title='商品', cost=200),  # Chinese
        Item(item_id=2, title='Produkt', cost=150),  # German
        Item(item_id=3, title='منتج', cost=180),  # Arabic
        Item(item_id=4, title='🎁 Gift', cost=50),  # Emoji
    ]

    # Verify all items are created successfully
    for item in items:
        assert len(item.title) > 0
        assert item.cost >= 0

    # Verify sorting still works with Unicode
    sorted_items = sorted(items)
    assert len(sorted_items) == len(items)
