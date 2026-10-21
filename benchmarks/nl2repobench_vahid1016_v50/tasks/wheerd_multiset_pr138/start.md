## Multiset Project Introduction and Goals

`multiset` is a small, dependency-free Python library providing a multiset (bag) implementation: an unordered collection of hashable elements where each element may occur multiple times. It mirrors the API of the builtin `set`/`frozenset` types while storing data internally as a `dict` mapping each element to its multiplicity, and it exposes dictionary-like operations as well.

The package offers three public classes organized in an inheritance hierarchy:

- `BaseMultiset`: abstract base holding all shared logic (construction, iteration, comparisons, set operations, dict-style views). It cannot be instantiated directly.
- `Multiset`: the mutable variant, adding in-place updates (`update`, `add`, `remove`, `discard`, `pop`, `clear`, etc.).
- `FrozenMultiset`: the immutable, hashable variant, analogous to `frozenset`.

Key goals:
- Support the full set operation suite (union, intersection, difference, symmetric difference, subset/superset tests, disjointness) with multiset semantics, i.e. multiplicities are combined with min/max/sum rules as appropriate.
- Support mixing multisets with built-in `set`/`frozenset` and with arbitrary iterables or mappings of element→multiplicity for the named method variants.
- Only positive multiplicities are allowed; elements whose multiplicity drops to zero (or below) are removed automatically, so the underlying dict never stores zero-count entries.
- Behave as a `Mapping`/`MutableMapping` (registered with `collections.abc` accordingly), supporting `items()`, `keys()`, `values()`, `get`, `pop`, and `len` equal to the total number of elements (sum of multiplicities).
- Be fully typed (ships a `py.typed` marker) and installable via setuptools with setuptools-scm versioning.

## Natural Language Instructions (Prompt)

1. Create a Python package named `multiset` containing a single module `multiset/__init__.py` that holds the entire implementation, plus a `multiset/py.typed` marker file (empty) so type checkers treat the package as typed.
2. Export exactly the names `BaseMultiset`, `Multiset`, and `FrozenMultiset` via `__all__` from the `multiset` package.
3. Implement `BaseMultiset` as the abstract base class:
   - Its `__new__` must raise `TypeError` with the message "Cannot instantiate BaseMultiset directly, use either Multiset or FrozenMultiset." when `BaseMultiset` itself is instantiated; both subclasses must be constructible.
   - Use `__slots__ = ('_elements', '_total')` where `_elements` is a `defaultdict(int)` mapping elements to positive multiplicities and `_total` is the cached sum of multiplicities.
   - Constructor signature `BaseMultiset(iterable=None)`. If `iterable` is another `BaseMultiset` (or subclass instance), copy its elements and total. If it is `None`, create an empty multiset. If it is a `dict` or other `Mapping`, keep only entries with multiplicity strictly greater than zero and add their multiplicities to the total. Otherwise treat it as an iterable of elements, counting each occurrence once (with a fast path when the object is `Sized` for computing the total). Elements do not need to be hashable-checked explicitly; unhashable elements simply raise naturally when inserted into the dict.
4. Implement the shared read-side protocol of `BaseMultiset` with the following observable semantics:
   - `__contains__(element)`: membership of a distinct element (multiplicity ≥ 1).
   - `__getitem__(element)`: return the element's multiplicity or 0 if absent.
   - `__len__`: the total number of elements (sum of multiplicities), not the number of distinct elements.
   - `__bool__`: True iff total > 0.
   - `__iter__`: iterate over each distinct element repeated according to its multiplicity (i.e. the total number of yielded items equals `len()`).
   - `__str__`: render as `{elem, elem, ...}` using the str of each iterated element; `__repr__`: render as `ClassName({elem!r: mult!r, ...})`.
   - `__getstate__`/`__setstate__` supporting pickling via a `(total, elements)` tuple.
   - `copy()` (also aliased as the copy-protocol `__copy__`): shallow copy returning an instance of the actual subclass.
   - `items()` returning `_elements.items()`, `distinct_elements()` returning `_elements.keys()`, and `multiplicities()` returning `_elements.values()`; `values` must be a plain alias of `multiplicities` (same object).
   - `get(element, default)` returning the multiplicity or the given default (default is a required positional argument).
   - Class method `from_elements(cls, elements, multiplicity)` building a multiset where every given distinct element has the given multiplicity (use `dict.fromkeys`).
5. Implement the non-mutating set operations as `BaseMultiset` methods:
   - `union(*others)`: elementwise maximum multiplicity; accepts any number of other iterables/mappings/multisets, each converted appropriately.
   - `combine(*others)`: elementwise sum of multiplicities.
   - `intersection(*others)`: elementwise minimum multiplicity (missing elements count as 0) across all others.
   - `difference(*others)`: subtract multiplicities of the others, flooring at removal when the result would be ≤ 0.
   - `symmetric_difference(other)`: exactly one argument; multiplicity is the absolute difference of multiplicities; result built fresh via `self.__class__()`.
   - `times(factor)`: multiply each multiplicity by a non-negative integer; factor 0 yields an empty multiset of the same class; negative factor raises `ValueError('The factor must no be negative.')`.
   - `isdisjoint(other)`: True iff no distinct element of the multiset occurs in `other`; `other` may be any iterable which is then converted, but is used directly if it is a `Container` (its `__contains__` is consulted).
   - `issubset(other)`, `issuperset(other)`: non-strict subset/superset tests; `other` may be any iterable or mapping and is converted to a multiset for the comparison.
6. Implement the corresponding operator dunders on `BaseMultiset`, each of which returns `NotImplemented` if the other operand is neither a `BaseMultiset`/`set`/`frozenset` nor (per the operand check used) an instance of `collections.abc.Set`:
   - `__or__`/`__ror__` → union; `__add__`/`__radd__` → combine; `__and__`/`__rand__` → intersection; `__xor__`/`__rxor__` → symmetric difference; `__sub__` → difference; `__rsub__` computes `other.multiset - self` by converting the left operand via `_as_multiset`.
   - `__mul__`/`__rmul__` → `times(factor)` only when `factor` is an `int`, otherwise `NotImplemented`.
   - Equality/comparison: `__eq__` and `__ne__` compare against other `BaseMultiset` instances by total and element map; against `set`/`frozenset` (or any `Set`) a multiset equals the set iff totals match and every distinct element of the multiset is in the set. Comparisons against non-`Set` types return `NotImplemented`. `__le__`/`__lt__`/`__ge__`/`__gt__` implement (strict/non-strict) subset and superset comparisons respectively, delegating to `issubset`/`issuperset` logic with a strict flag.
7. Implement private helpers `_as_multiset(other)` and `_as_mapping(iterable)` as class/static methods used to coerce operands: `_as_multiset` returns `BaseMultiset` instances and non-iterables raise `TypeError("'%s' object is not iterable" % type(other))`, otherwise constructs `cls(other)`; `_as_mapping` returns the element dict of a multiset, a `dict` or any `Mapping` directly, or counts occurrences into a fresh `dict` for plain iterables, raising the same `TypeError` for non-i