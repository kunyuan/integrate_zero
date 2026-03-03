# IntegrateZero Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a full end-to-end IntegrateZero demo: supervised pretraining -> MCTS search -> RL self-play for symbolic integration via multi-step expression transformation.

**Architecture:** Decoder-only Transformer (~12M params) generates candidate expressions autoregressively. MCTS explores multi-step transformation paths. SymPy derivative verification serves as the perfect reward signal. Prefix notation represents all expressions.

**Tech Stack:** Python 3.8+, PyTorch, SymPy, pyproject.toml

**Design Doc:** `docs/plans/2026-03-03-integrate-zero-demo-design.md`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `integrate_zero/__init__.py`
- Create: `integrate_zero/data/__init__.py`
- Create: `integrate_zero/model/__init__.py`
- Create: `integrate_zero/mcts/__init__.py`
- Create: `integrate_zero/train/__init__.py`
- Create: `integrate_zero/eval/__init__.py`
- Create: `integrate_zero/verify/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "integrate-zero"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0",
    "sympy>=1.12",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create package directories and __init__.py files**

All `__init__.py` files are empty initially.

**Step 3: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: installs successfully

**Step 4: Verify pytest discovers tests directory**

Run: `pytest --collect-only`
Expected: "no tests ran" (no tests yet, but no errors)

**Step 5: Commit**

```bash
git init
git add pyproject.toml integrate_zero/ tests/
git commit -m "feat: initial project scaffolding"
```

---

### Task 2: Vocabulary & Tokenizer

**Files:**
- Create: `integrate_zero/data/vocabulary.py`
- Create: `tests/test_vocabulary.py`

**Step 1: Write failing tests**

```python
# tests/test_vocabulary.py
from integrate_zero.data.vocabulary import Vocabulary

def test_special_tokens_have_fixed_ids():
    vocab = Vocabulary()
    assert vocab.token_to_id("PAD") == 0
    assert vocab.token_to_id("BOS") == 1
    assert vocab.token_to_id("EOS") == 2
    assert vocab.token_to_id("SEP") == 3

def test_operators_in_vocab():
    vocab = Vocabulary()
    for op in ["add", "sub", "mul", "div", "pow", "neg"]:
        assert vocab.token_to_id(op) is not None

def test_functions_in_vocab():
    vocab = Vocabulary()
    for fn in ["sin", "cos", "tan", "exp", "log", "sqrt",
               "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh"]:
        assert vocab.token_to_id(fn) is not None

def test_int_token():
    vocab = Vocabulary()
    assert vocab.token_to_id("INT") is not None

def test_variables_and_params():
    vocab = Vocabulary()
    assert vocab.token_to_id("x") is not None
    for p in ["a", "b", "c", "d"]:
        assert vocab.token_to_id(p) is not None
    for p in ["k", "l", "m", "n"]:
        assert vocab.token_to_id(p) is not None

def test_numeric_constants():
    vocab = Vocabulary()
    for i in range(-10, 11):
        assert vocab.token_to_id(str(i)) is not None
    assert vocab.token_to_id("pi") is not None
    assert vocab.token_to_id("e") is not None

def test_roundtrip():
    vocab = Vocabulary()
    for tok in ["add", "sin", "x", "INT", "3", "pi"]:
        assert vocab.id_to_token(vocab.token_to_id(tok)) == tok

def test_arity():
    vocab = Vocabulary()
    assert vocab.arity("add") == 2
    assert vocab.arity("sin") == 1
    assert vocab.arity("INT") == 2
    assert vocab.arity("x") == 0
    assert vocab.arity("3") == 0
    assert vocab.arity("pi") == 0

def test_vocab_size():
    vocab = Vocabulary()
    assert 80 <= len(vocab) <= 100
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vocabulary.py -v`
Expected: FAIL (module not found)

**Step 3: Implement Vocabulary**

```python
# integrate_zero/data/vocabulary.py

# Arity lookup: how many children each token takes in prefix notation
_ARITIES = {
    # operators (arity 2)
    "add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2,
    # unary
    "neg": 1,
    # functions (arity 1)
    "sin": 1, "cos": 1, "tan": 1, "exp": 1, "log": 1, "sqrt": 1,
    "arcsin": 1, "arccos": 1, "arctan": 1,
    "sinh": 1, "cosh": 1, "tanh": 1,
    # INT: (integrand, variable)
    "INT": 2,
}

_SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "SEP"]

_VARIABLES = ["x"]
_REAL_PARAMS = ["a", "b", "c", "d"]
_INTEGER_PARAMS = ["k", "l", "m", "n"]
_NUMERIC_CONSTS = [str(i) for i in range(-10, 11)] + ["pi", "e"]


class Vocabulary:
    def __init__(self):
        tokens = (
            _SPECIAL_TOKENS
            + list(_ARITIES.keys())
            + _VARIABLES
            + _REAL_PARAMS
            + _INTEGER_PARAMS
            + _NUMERIC_CONSTS
        )
        self._tok2id = {tok: i for i, tok in enumerate(tokens)}
        self._id2tok = {i: tok for tok, i in self._tok2id.items()}

    def token_to_id(self, token: str) -> int | None:
        return self._tok2id.get(token)

    def id_to_token(self, id: int) -> str | None:
        return self._id2tok.get(id)

    def arity(self, token: str) -> int:
        return _ARITIES.get(token, 0)

    def __len__(self):
        return len(self._tok2id)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vocabulary.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/data/vocabulary.py tests/test_vocabulary.py
git commit -m "feat: vocabulary and tokenizer with arity tracking"
```

---

### Task 3: Prefix Expression <-> SymPy Conversion

**Files:**
- Create: `integrate_zero/data/prefix.py`
- Create: `tests/test_prefix.py`

**Step 1: Write failing tests**

```python
# tests/test_prefix.py
import sympy
from integrate_zero.data.prefix import sympy_to_prefix, prefix_to_sympy

x = sympy.Symbol("x")
a = sympy.Symbol("a", real=True)
n = sympy.Symbol("n", integer=True)

def test_simple_variable():
    assert sympy_to_prefix(x) == ["x"]
    assert prefix_to_sympy(["x"]) == x

def test_integer_constant():
    assert sympy_to_prefix(sympy.Integer(3)) == ["3"]
    assert prefix_to_sympy(["3"]) == sympy.Integer(3)

def test_addition():
    expr = x + sympy.Integer(1)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["add", "x", "1"]
    assert prefix_to_sympy(tokens) == expr

def test_sin():
    expr = sympy.sin(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["sin", "x"]
    assert prefix_to_sympy(tokens) == expr

def test_nested():
    # sin(x^2 + 1)
    expr = sympy.sin(x**2 + 1)
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0

def test_with_integral():
    # INT sin(x) dx
    expr = sympy.Integral(sympy.sin(x), x)
    tokens = sympy_to_prefix(expr)
    assert tokens[0] == "INT"
    assert tokens[-1] == "x"  # integration variable
    assert prefix_to_sympy(tokens) == expr

def test_with_parameter():
    # a * x^n
    expr = a * x**n
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0

def test_roundtrip_complex():
    # x*sin(x) + cos(x)
    expr = x * sympy.sin(x) + sympy.cos(x)
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0

def test_negation():
    expr = -x
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0

def test_pi_and_e():
    expr = sympy.pi * x + sympy.E
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prefix.py -v`
Expected: FAIL

**Step 3: Implement prefix.py**

This is the most subtle module. Key considerations:
- SymPy `Add` and `Mul` are n-ary but our prefix is binary — need to left-fold: `Add(a,b,c)` -> `["add", "add", "a", "b", "c"]`
- SymPy represents `-x` as `Mul(-1, x)` — detect and emit `["neg", "x"]`
- SymPy represents `x - y` as `Add(x, Mul(-1, y))` — emit `["sub", ...]`
- `sympy.Integral(f, x)` maps to `["INT", ...f_tokens..., "x"]`
- `prefix_to_sympy` uses a stack-based parser consuming tokens left to right

Implementation will handle these SymPy expression types:
- `sympy.Symbol` -> variable/parameter token
- `sympy.Integer`, `sympy.Rational` -> numeric tokens
- `sympy.pi`, `sympy.E` -> `"pi"`, `"e"`
- `sympy.Add` -> binary `"add"` / `"sub"` chain
- `sympy.Mul` -> binary `"mul"` / `"neg"` chain
- `sympy.Pow` -> `"pow"`
- `sympy.sin`, etc. -> function tokens
- `sympy.Integral` -> `"INT"`

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prefix.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/data/prefix.py tests/test_prefix.py
git commit -m "feat: bidirectional prefix notation <-> SymPy conversion"
```

---

### Task 4: Verifier

**Files:**
- Create: `integrate_zero/verify/verifier.py`
- Create: `tests/test_verifier.py`

**Step 1: Write failing tests**

```python
# tests/test_verifier.py
import sympy
from integrate_zero.verify.verifier import verify_step, is_terminal, StepType

x = sympy.Symbol("x")

def test_valid_integration_step():
    # INT x*cos(x) dx -> x*sin(x) + cos(x)  (terminal)
    A = sympy.Integral(x * sympy.cos(x), x)
    B = x * sympy.sin(x) + sympy.cos(x)
    result = verify_step(A, B)
    assert result == StepType.INTEGRATION

def test_valid_partial_integration():
    # INT x*cos(x) dx -> x*sin(x) - INT sin(x) dx  (non-terminal)
    A = sympy.Integral(x * sympy.cos(x), x)
    B = x * sympy.sin(x) - sympy.Integral(sympy.sin(x), x)
    result = verify_step(A, B)
    assert result == StepType.INTEGRATION

def test_identity_rewrite():
    # INT sin(x)*cos(x) dx -> INT sin(2x)/2 dx
    A = sympy.Integral(sympy.sin(x) * sympy.cos(x), x)
    B = sympy.Integral(sympy.sin(2*x) / 2, x)
    result = verify_step(A, B)
    assert result == StepType.IDENTITY

def test_invalid_step():
    # INT sin(x) dx -> x^2 (wrong)
    A = sympy.Integral(sympy.sin(x), x)
    B = x**2
    result = verify_step(A, B)
    assert result == StepType.INVALID

def test_is_terminal():
    assert is_terminal(x * sympy.sin(x) + sympy.cos(x)) == True
    assert is_terminal(x * sympy.sin(x) - sympy.Integral(sympy.sin(x), x)) == False
    assert is_terminal(sympy.Integral(x, x)) == False

def test_numerical_fallback():
    # Use an expression where sympy.simplify might struggle but numerical check works
    A = sympy.Integral(sympy.exp(x) * sympy.cos(x), x)
    B = sympy.exp(x) * (sympy.sin(x) + sympy.cos(x)) / 2
    result = verify_step(A, B)
    assert result == StepType.INTEGRATION
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verifier.py -v`
Expected: FAIL

**Step 3: Implement verifier**

```python
# integrate_zero/verify/verifier.py
import enum
import sympy
import random

class StepType(enum.Enum):
    INTEGRATION = "integration"
    IDENTITY = "identity"
    INVALID = "invalid"

def _symbolic_equal(expr1, expr2, x=sympy.Symbol("x")) -> bool:
    """Check if two expressions are symbolically equal."""
    try:
        diff = sympy.simplify(expr1 - expr2)
        if diff == 0:
            return True
    except Exception:
        pass
    return _numerical_equal(expr1, expr2, x)

def _numerical_equal(expr1, expr2, x=sympy.Symbol("x"),
                     n_points=20, tol=1e-8) -> bool:
    """Numerical fallback: sample random points and compare."""
    free = expr1.free_symbols | expr2.free_symbols
    for _ in range(n_points):
        subs = {s: random.uniform(-5, 5) for s in free}
        try:
            v1 = complex(expr1.subs(subs))
            v2 = complex(expr2.subs(subs))
            if abs(v1 - v2) > tol:
                return False
        except Exception:
            continue
    return True

def is_terminal(expr) -> bool:
    """Check if expression contains no Integral symbols."""
    return not expr.has(sympy.Integral)

def verify_step(A, B, x=sympy.Symbol("x")) -> StepType:
    """Verify a transformation step from A to B.

    A is the current state (may contain Integral).
    B is the proposed next state.
    """
    # Check identity rewrite: A == B algebraically
    if _symbolic_equal(A, B, x):
        return StepType.IDENTITY

    # Check integration step: extract integrands from A,
    # verify by differentiating non-integral parts of B
    # For terminal B (no INT): check d/dx(B) == integrand of A
    # For non-terminal B (has INT): more complex partial verification
    if is_terminal(B):
        # A should be Integral(f, x) possibly with extra terms
        # Check d/dx(B) equals the original integrand
        integrand = _extract_full_integrand(A)
        if integrand is not None:
            dB = sympy.diff(B, x)
            if _symbolic_equal(dB, integrand, x):
                return StepType.INTEGRATION
    else:
        # Partial integration: A and B both have Integral parts
        # Verify: the non-integral parts are consistent
        # Strategy: diff the non-integral parts and check consistency
        # Simplified approach: numerical verification of the full expressions
        # Treat both as functions of x and check they represent the same antiderivative
        A_eval = _try_evaluate_integral(A)
        B_eval = _try_evaluate_integral(B)
        if A_eval is not None and B_eval is not None:
            if _symbolic_equal(A_eval, B_eval, x):
                return StepType.INTEGRATION

    return StepType.INVALID

def _extract_full_integrand(expr):
    """Extract the integrand from an expression that is purely Integral(f, x)."""
    if isinstance(expr, sympy.Integral):
        return expr.args[0]
    return None

def _try_evaluate_integral(expr):
    """Try to evaluate any Integral subexpressions using SymPy."""
    try:
        return expr.doit()
    except Exception:
        return None
```

Note: The verifier implementation above is a starting point. The partial integration verification (non-terminal B) is tricky — the `doit()` approach works when SymPy can solve the remaining integrals, which is a reasonable assumption for verification. During implementation, iterate on edge cases found during testing.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verifier.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/verify/verifier.py tests/test_verifier.py
git commit -m "feat: step verifier with derivative checking and numerical fallback"
```

---

### Task 5: Random Expression Generator

**Files:**
- Create: `integrate_zero/data/generator.py`
- Create: `tests/test_generator.py`

**Step 1: Write failing tests**

```python
# tests/test_generator.py
import sympy
from integrate_zero.data.generator import generate_expression, generate_training_pair
from integrate_zero.data.prefix import sympy_to_prefix, prefix_to_sympy

x = sympy.Symbol("x")

def test_generate_expression_returns_sympy():
    expr = generate_expression(max_depth=3)
    assert isinstance(expr, sympy.Basic)
    assert x in expr.free_symbols

def test_generate_expression_depth():
    # Low depth should produce simple expressions
    expr = generate_expression(max_depth=2)
    tokens = sympy_to_prefix(expr)
    assert len(tokens) <= 15  # reasonable upper bound for depth-2

def test_generate_training_pair():
    f, F = generate_training_pair(max_depth=3)
    # f should be the derivative of F
    dF = sympy.diff(F, x)
    assert sympy.simplify(dF - f) == 0

def test_generate_multiple_unique():
    pairs = [generate_training_pair(max_depth=3) for _ in range(20)]
    # At least some should be different (probabilistic but very likely)
    f_strs = [str(p[0]) for p in pairs]
    assert len(set(f_strs)) > 1

def test_generate_expression_roundtrips_through_prefix():
    for _ in range(10):
        expr = generate_expression(max_depth=3)
        tokens = sympy_to_prefix(expr)
        reconstructed = prefix_to_sympy(tokens)
        assert sympy.simplify(reconstructed - expr) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generator.py -v`
Expected: FAIL

**Step 3: Implement generator**

The generator randomly builds expression trees:
- Leaf nodes: `x`, numeric constants, parameters
- Internal nodes: operators and functions
- `max_depth` controls tree depth
- Rejects expressions that can't roundtrip through prefix notation

```python
# integrate_zero/data/generator.py
import random
import sympy

x = sympy.Symbol("x")
_REAL_PARAMS = [sympy.Symbol(s, real=True) for s in ["a", "b", "c", "d"]]
_INT_PARAMS = [sympy.Symbol(s, integer=True) for s in ["k", "l", "m", "n"]]

_UNARY_FNS = [sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log,
              sympy.sqrt, sympy.asin, sympy.acos, sympy.atan,
              sympy.sinh, sympy.cosh, sympy.tanh]
_BINARY_OPS = [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
]

def generate_expression(max_depth: int, _depth: int = 0) -> sympy.Expr:
    """Generate a random symbolic expression containing x."""
    if _depth >= max_depth or (_depth > 0 and random.random() < 0.3):
        return _random_leaf()

    if random.random() < 0.3:
        # unary function
        fn = random.choice(_UNARY_FNS)
        child = generate_expression(max_depth, _depth + 1)
        return fn(child)
    else:
        # binary operator
        op = random.choice(_BINARY_OPS)
        left = generate_expression(max_depth, _depth + 1)
        right = generate_expression(max_depth, _depth + 1)
        return op(left, right)

def _random_leaf():
    """Random leaf: x, integer constant, or parameter."""
    r = random.random()
    if r < 0.5:
        return x
    elif r < 0.8:
        return sympy.Integer(random.randint(-5, 5))
    else:
        return random.choice(_REAL_PARAMS + _INT_PARAMS)

def generate_training_pair(max_depth: int = 4):
    """Generate (f, F) where f = dF/dx.

    F is a random expression, f is its derivative.
    Retries if f is trivially zero.
    """
    for _ in range(100):
        F = generate_expression(max_depth)
        f = sympy.diff(F, x)
        if f != 0 and x in f.free_symbols:
            return f, F
    # Fallback
    F = x ** 2
    return sympy.diff(F, x), F
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generator.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/data/generator.py tests/test_generator.py
git commit -m "feat: random expression generator and training pair generation"
```

---

### Task 6: PyTorch Dataset

**Files:**
- Create: `integrate_zero/data/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Write failing tests**

```python
# tests/test_dataset.py
import torch
from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.data.vocabulary import Vocabulary

def test_dataset_length():
    ds = IntegrationDataset(num_samples=100, max_depth=3)
    assert len(ds) == 100

def test_dataset_item_shape():
    vocab = Vocabulary()
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    item = ds[0]
    assert "input_ids" in item
    assert "target_ids" in item
    assert "value_label" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["target_ids"], torch.Tensor)
    assert item["input_ids"].dtype == torch.long
    assert item["target_ids"].dtype == torch.long

def test_dataset_input_starts_with_bos():
    vocab = Vocabulary()
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    item = ds[0]
    assert item["input_ids"][0].item() == vocab.token_to_id("BOS")

def test_dataset_has_sep_token():
    vocab = Vocabulary()
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    item = ds[0]
    sep_id = vocab.token_to_id("SEP")
    assert sep_id in item["input_ids"].tolist()

def test_collate_pads_to_same_length():
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape[1] > 0  # padded to max length
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL

**Step 3: Implement dataset**

The dataset generates (f, F) pairs offline, tokenizes them into prefix notation, and encodes as `[BOS] f_tokens [SEP] F_tokens [EOS]`.

```python
# integrate_zero/data/dataset.py
import torch
from torch.utils.data import Dataset
from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.data.generator import generate_training_pair
from integrate_zero.data.prefix import sympy_to_prefix

class IntegrationDataset(Dataset):
    def __init__(self, num_samples: int, max_depth: int = 4, max_len: int = 128):
        self.vocab = Vocabulary()
        self.max_len = max_len
        self.samples = []
        for _ in range(num_samples):
            try:
                f, F = generate_training_pair(max_depth)
                f_tokens = sympy_to_prefix(f)
                F_tokens = sympy_to_prefix(F)
                self.samples.append((f_tokens, F_tokens))
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_tokens, F_tokens = self.samples[idx]
        # [BOS] f_tokens [SEP] F_tokens [EOS]
        full_tokens = ["BOS"] + f_tokens + ["SEP"] + F_tokens + ["EOS"]
        ids = [self.vocab.token_to_id(t) for t in full_tokens]

        # input_ids: all tokens (for causal LM, input = target shifted)
        input_ids = torch.tensor(ids, dtype=torch.long)

        # target_ids: -100 for f_tokens part (no loss), actual ids for F_tokens part
        sep_pos = len(f_tokens) + 1  # +1 for BOS
        target_ids = torch.full_like(input_ids, -100)
        target_ids[sep_pos + 1:] = input_ids[sep_pos + 1:]  # predict after SEP

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "value_label": torch.tensor(1.0),  # all generated pairs are solvable
        }

    def collate_fn(self, batch):
        pad_id = self.vocab.token_to_id("PAD")
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        target_ids = torch.full((len(batch), max_len), -100, dtype=torch.long)
        value_labels = torch.zeros(len(batch))

        for i, item in enumerate(batch):
            L = item["input_ids"].size(0)
            input_ids[i, :L] = item["input_ids"]
            target_ids[i, :L] = item["target_ids"]
            value_labels[i] = item["value_label"]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "value_labels": value_labels,
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/data/dataset.py tests/test_dataset.py
git commit -m "feat: PyTorch dataset with tokenization and collation"
```

---

### Task 7: Transformer Model + Value Head

**Files:**
- Create: `integrate_zero/model/transformer.py`
- Create: `tests/test_model.py`

**Step 1: Write failing tests**

```python
# tests/test_model.py
import torch
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary

def test_model_forward_shape():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    input_ids = torch.randint(0, len(vocab), (2, 20))
    logits, value = model(input_ids, sep_positions=torch.tensor([5, 5]))
    assert logits.shape == (2, 20, len(vocab))
    assert value.shape == (2,)
    assert (value >= 0).all() and (value <= 1).all()

def test_model_generate():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    # Input: [BOS] sin x [SEP]
    bos = vocab.token_to_id("BOS")
    sep = vocab.token_to_id("SEP")
    sin_id = vocab.token_to_id("sin")
    x_id = vocab.token_to_id("x")
    prompt = torch.tensor([[bos, sin_id, x_id, sep]])
    generated = model.generate(prompt, max_new_tokens=10, vocab=vocab)
    assert generated.shape[0] == 1
    assert generated.shape[1] > 4  # prompt + at least some generated tokens

def test_model_param_count():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=384, nhead=6,
                               num_layers=8, d_ff=1536)
    total = sum(p.numel() for p in model.parameters())
    assert 10_000_000 < total < 20_000_000  # ~12M expected
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL

**Step 3: Implement model**

```python
# integrate_zero/model/transformer.py
import torch
import torch.nn as nn
import math

class IntegrateZeroModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 384, nhead: int = 6,
                 num_layers: int = 8, d_ff: int = 1536, max_seq_len: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        # Use TransformerDecoder as a causal decoder-only stack
        # (no encoder, pass dummy memory of zeros)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor,
                sep_positions: torch.Tensor | None = None):
        B, T = input_ids.shape
        device = input_ids.device

        tok_emb = self.embedding(input_ids)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        h = self.dropout(tok_emb + pos_emb)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Pad mask
        pad_mask = (input_ids == 0)  # PAD = 0

        # Dummy memory (decoder-only: no encoder)
        memory = torch.zeros(B, 1, self.d_model, device=device)

        h = self.decoder(
            h, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )
        h = self.ln_f(h)

        logits = self.lm_head(h)

        # Value: extract hidden state at SEP position
        if sep_positions is not None:
            sep_h = h[torch.arange(B), sep_positions]
        else:
            sep_h = h[:, 0]  # fallback: use first token
        value = self.value_head(sep_h).squeeze(-1)

        return logits, value

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                 vocab=None, temperature: float = 1.0):
        """Autoregressive generation with optional arity-based constraint."""
        self.eval()
        generated = prompt.clone()
        for _ in range(max_new_tokens):
            logits, _ = self(generated)
            next_logits = logits[:, -1, :] / temperature

            # Arity constraint masking (if vocab provided)
            if vocab is not None:
                next_logits = self._apply_arity_mask(next_logits, generated, vocab)

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS
            eos_id = vocab.token_to_id("EOS") if vocab else 2
            if (next_token == eos_id).all():
                break
        return generated

    def _apply_arity_mask(self, logits, generated, vocab):
        """Mask illegal tokens based on arity stack state."""
        # Implementation tracks arity stack for each sequence in batch
        # to enforce syntactically valid prefix expressions.
        # This is called during generation only.
        # Detailed implementation in generation — for now return logits unchanged.
        return logits
```

Note: `_apply_arity_mask` is a placeholder — full arity-stack masking will be refined during Task 9 (MCTS) when generation is used in the search loop.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/model/transformer.py tests/test_model.py
git commit -m "feat: decoder-only transformer with value head"
```

---

### Task 8: Arity-Constrained Decoding

**Files:**
- Create: `integrate_zero/model/decoding.py`
- Modify: `integrate_zero/model/transformer.py` (wire in arity mask)
- Create: `tests/test_decoding.py`

**Step 1: Write failing tests**

```python
# tests/test_decoding.py
from integrate_zero.model.decoding import ArityMask
from integrate_zero.data.vocabulary import Vocabulary

def test_arity_mask_forces_eos_when_stack_empty():
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    # After generating a complete expression like "x", stack is empty
    tokens = ["BOS", "x", "SEP", "x"]
    allowed = mask.get_allowed_tokens(tokens[2:])  # after SEP
    eos_id = vocab.token_to_id("EOS")
    assert eos_id in allowed

def test_arity_mask_blocks_eos_when_stack_not_empty():
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    # After "add x", we still need one more child
    tokens = ["add", "x"]
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id not in allowed

def test_arity_mask_after_binary_op():
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    # After "add", need 2 children -> everything except EOS/SEP/PAD/BOS
    tokens = ["add"]
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id not in allowed
    assert vocab.token_to_id("x") in allowed
    assert vocab.token_to_id("sin") in allowed

def test_complete_expression_valid():
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    # "add sin x x" is complete: add(sin(x), x)
    tokens = ["add", "sin", "x", "x"]
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id in allowed
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_decoding.py -v`
Expected: FAIL

**Step 3: Implement ArityMask**

```python
# integrate_zero/model/decoding.py
from integrate_zero.data.vocabulary import Vocabulary

class ArityMask:
    """Tracks arity stack to enforce valid prefix expression generation."""

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def get_allowed_tokens(self, generated_tokens: list[str]) -> set[int]:
        """Return set of allowed token IDs given tokens generated so far (after SEP)."""
        remaining = self._compute_remaining_slots(generated_tokens)

        allowed = set()
        for tok_id in range(len(self.vocab)):
            tok = self.vocab.id_to_token(tok_id)
            if tok in ("PAD", "BOS", "SEP"):
                continue
            if tok == "EOS":
                if remaining == 0:
                    allowed.add(tok_id)
                continue
            arity = self.vocab.arity(tok)
            if remaining > 0:
                allowed.add(tok_id)
        return allowed

    def _compute_remaining_slots(self, tokens: list[str]) -> int:
        """Compute how many more leaf tokens are needed to complete the expression."""
        if not tokens:
            return 1  # need at least one token
        remaining = 1  # start: need one expression
        for tok in tokens:
            remaining -= 1  # this token fills one slot
            remaining += self.vocab.arity(tok)  # but opens new slots
        return remaining
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_decoding.py -v`
Expected: all PASS

**Step 5: Wire ArityMask into transformer.py's _apply_arity_mask**

Update `_apply_arity_mask` in `integrate_zero/model/transformer.py` to use `ArityMask`.

**Step 6: Commit**

```bash
git add integrate_zero/model/decoding.py integrate_zero/model/transformer.py tests/test_decoding.py
git commit -m "feat: arity-constrained decoding for valid prefix expressions"
```

---

### Task 9: Supervised Training Loop (Phase 1)

**Files:**
- Create: `integrate_zero/train/supervised.py`
- Create: `tests/test_supervised.py`

**Step 1: Write failing test**

```python
# tests/test_supervised.py
import torch
from integrate_zero.train.supervised import SupervisedTrainer
from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary

def test_supervised_one_step():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    trainer = SupervisedTrainer(model, ds, vocab, batch_size=4, lr=1e-3)
    loss_before = trainer.evaluate_loss()
    trainer.train_epoch()
    loss_after = trainer.evaluate_loss()
    # Loss should decrease after one epoch on small dataset
    assert loss_after < loss_before

def test_supervised_saves_checkpoint(tmp_path):
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    trainer = SupervisedTrainer(model, ds, vocab, batch_size=4, lr=1e-3)
    trainer.train_epoch()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(path))
    assert path.exists()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_supervised.py -v`
Expected: FAIL

**Step 3: Implement supervised trainer**

```python
# integrate_zero/train/supervised.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from integrate_zero.data.vocabulary import Vocabulary

class SupervisedTrainer:
    def __init__(self, model, dataset, vocab: Vocabulary,
                 batch_size: int = 256, lr: float = 1e-4):
        self.model = model
        self.dataset = dataset
        self.vocab = vocab
        self.device = next(model.parameters()).device
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=dataset.collate_fn)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.policy_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.value_criterion = nn.BCELoss()

    def train_epoch(self):
        self.model.train()
        for batch in self.loader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            value_labels = batch["value_labels"].to(self.device)

            # Find SEP positions
            sep_id = self.vocab.token_to_id("SEP")
            sep_positions = (input_ids == sep_id).long().argmax(dim=1)

            logits, value = self.model(input_ids, sep_positions)

            # Policy loss: predict next token (shifted)
            policy_loss = self.policy_criterion(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                target_ids[:, 1:].reshape(-1)
            )

            # Value loss
            value_loss = self.value_criterion(value, value_labels)

            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate_loss(self):
        self.model.eval()
        total_loss = 0
        count = 0
        for batch in self.loader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            value_labels = batch["value_labels"].to(self.device)

            sep_id = self.vocab.token_to_id("SEP")
            sep_positions = (input_ids == sep_id).long().argmax(dim=1)

            logits, value = self.model(input_ids, sep_positions)
            policy_loss = self.policy_criterion(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                target_ids[:, 1:].reshape(-1)
            )
            value_loss = self.value_criterion(value, value_labels)
            total_loss += (policy_loss + value_loss).item()
            count += 1
        return total_loss / max(count, 1)

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_supervised.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/train/supervised.py tests/test_supervised.py
git commit -m "feat: supervised training loop with policy and value losses"
```

---

### Task 10: MCTS

**Files:**
- Create: `integrate_zero/mcts/search.py`
- Create: `tests/test_mcts.py`

**Step 1: Write failing tests**

```python
# tests/test_mcts.py
import sympy
from unittest.mock import MagicMock
from integrate_zero.mcts.search import MCTSNode, MCTS

x = sympy.Symbol("x")

def test_mcts_node_creation():
    expr = sympy.Integral(sympy.sin(x), x)
    node = MCTSNode(state=expr, prior=1.0)
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert len(node.children) == 0

def test_mcts_node_ucb():
    parent = MCTSNode(state=None, prior=1.0)
    parent.visit_count = 10
    child = MCTSNode(state=None, prior=0.5, parent=parent)
    child.visit_count = 2
    child.value_sum = 1.0
    ucb = child.ucb_score(c_puct=1.5)
    assert isinstance(ucb, float)
    assert ucb > 0

def test_mcts_finds_simple_integral():
    """MCTS should solve INT sin(x) dx = -cos(x) with a mock model."""
    # Mock model that always proposes -cos(x) as candidate
    mock_model = MagicMock()
    problem = sympy.Integral(sympy.sin(x), x)

    mcts = MCTS(model=mock_model, max_steps=20, num_candidates=8,
                search_budget=200, c_puct=1.5)

    # For this test, we directly test the node expansion and verification logic
    # by providing pre-generated candidates
    root = MCTSNode(state=problem, prior=1.0)
    candidate = -sympy.cos(x)
    child = root.expand_with(candidate, prior=0.5)
    assert child is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcts.py -v`
Expected: FAIL

**Step 3: Implement MCTS**

```python
# integrate_zero/mcts/search.py
import math
import sympy
from integrate_zero.verify.verifier import verify_step, is_terminal, StepType

class MCTSNode:
    def __init__(self, state, prior: float = 1.0, parent=None, step_type=None):
        self.state = state              # SymPy expression
        self.prior = prior              # P(action) from policy
        self.parent = parent
        self.step_type = step_type      # StepType of the action that led here
        self.children: list[MCTSNode] = []
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 1.5) -> float:
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def expand_with(self, candidate_expr, prior: float) -> "MCTSNode | None":
        step_type = verify_step(self.state, candidate_expr)
        if step_type == StepType.INVALID:
            return None
        child = MCTSNode(state=candidate_expr, prior=prior,
                         parent=self, step_type=step_type)
        self.children.append(child)
        return child

    def backpropagate(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def best_child(self, c_puct: float = 1.5) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def is_terminal(self) -> bool:
        return is_terminal(self.state)


class MCTS:
    def __init__(self, model, max_steps: int = 20, num_candidates: int = 8,
                 search_budget: int = 200, c_puct: float = 1.5):
        self.model = model
        self.max_steps = max_steps
        self.num_candidates = num_candidates
        self.search_budget = search_budget
        self.c_puct = c_puct

    def search(self, problem) -> list | None:
        """Run MCTS from problem. Returns solution trajectory or None."""
        root = MCTSNode(state=problem, prior=1.0)
        trajectory = []

        for step in range(self.max_steps):
            # Run search iterations at current node
            best = self._run_search(root)
            if best is None:
                return None  # no valid moves

            trajectory.append(best.state)

            if best.is_terminal():
                return trajectory  # solved

            # Move root to best child for next step
            root = best

        return None  # timeout

    def _run_search(self, root: MCTSNode) -> MCTSNode | None:
        """Run search iterations from root, return best child."""
        for _ in range(self.search_budget // self.max_steps):
            # Select leaf
            leaf = self._select(root)

            # Expand: generate candidates from model
            if not leaf.children:
                self._expand(leaf)

            # Evaluate and backpropagate
            if leaf.children:
                for child in leaf.children:
                    value = self._evaluate(child)
                    child.backpropagate(value)

        if not root.children:
            return None
        # Return most-visited child
        return max(root.children, key=lambda c: c.visit_count)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            node = node.best_child(self.c_puct)
        return node

    def _expand(self, node: MCTSNode):
        """Generate candidate expressions using the model and expand the node."""
        candidates = self._generate_candidates(node.state)
        for expr, prior in candidates:
            node.expand_with(expr, prior)

    def _generate_candidates(self, state) -> list[tuple]:
        """Use the model to generate candidate next expressions.

        Returns list of (sympy_expr, prior_probability) tuples.
        """
        # This method bridges the model (token generation) with MCTS (SymPy expressions).
        # 1. Convert state to prefix tokens
        # 2. Encode as [BOS] state_tokens [SEP]
        # 3. Sample num_candidates completions from model
        # 4. Convert each back to SymPy
        # 5. Return with their probabilities as priors

        # Implementation depends on model's generate() method
        # Placeholder — will be wired during integration
        return []

    def _evaluate(self, node: MCTSNode) -> float:
        """Use value head to estimate win probability from this state."""
        if node.is_terminal():
            return 1.0
        # Use model's value head
        # Placeholder — will be wired during integration
        return 0.5
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcts.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/mcts/search.py tests/test_mcts.py
git commit -m "feat: MCTS search with UCB selection and backpropagation"
```

---

### Task 11: MCTS-Model Integration

**Files:**
- Modify: `integrate_zero/mcts/search.py` (fill in `_generate_candidates` and `_evaluate`)
- Create: `tests/test_mcts_integration.py`

**Step 1: Write failing test**

```python
# tests/test_mcts_integration.py
import torch
import sympy
from integrate_zero.mcts.search import MCTS
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary

x = sympy.Symbol("x")

def test_mcts_generates_candidates():
    """Model generates candidate expressions that are parseable."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    mcts = MCTS(model=model, vocab=vocab, num_candidates=4, search_budget=20)
    problem = sympy.Integral(x**2, x)
    candidates = mcts._generate_candidates(problem)
    # At least some candidates should be parseable (with random model, some may fail)
    assert isinstance(candidates, list)

def test_mcts_evaluate_returns_float():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    mcts = MCTS(model=model, vocab=vocab)
    from integrate_zero.mcts.search import MCTSNode
    node = MCTSNode(state=x**2, prior=1.0)
    val = mcts._evaluate(node)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcts_integration.py -v`
Expected: FAIL

**Step 3: Implement _generate_candidates and _evaluate**

Fill in the placeholder methods in `MCTS`:
- `_generate_candidates`: converts state to tokens, runs model.generate() with sampling, converts back to SymPy
- `_evaluate`: converts state to tokens, runs model forward pass, extracts value head output

Update `MCTS.__init__` to accept `vocab` parameter.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcts_integration.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/mcts/search.py tests/test_mcts_integration.py
git commit -m "feat: wire model generation and value estimation into MCTS"
```

---

### Task 12: RL Training Loop (Phase 2)

**Files:**
- Create: `integrate_zero/train/rl.py`
- Create: `tests/test_rl.py`

**Step 1: Write failing test**

```python
# tests/test_rl.py
import torch
import sympy
from integrate_zero.train.rl import RLTrainer
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.data.generator import generate_training_pair

x = sympy.Symbol("x")

def test_rl_collect_trajectory():
    """RL trainer can attempt to collect a trajectory (may fail on random model)."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    trainer = RLTrainer(model, vocab, num_candidates=4, search_budget=10)
    problem = sympy.Integral(x**2, x)
    result = trainer.collect_episode(problem)
    assert "trajectory" in result
    assert "solved" in result
    assert isinstance(result["solved"], bool)

def test_rl_train_step_runs():
    """RL train step doesn't crash (may not improve with tiny model)."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    trainer = RLTrainer(model, vocab, num_candidates=4, search_budget=10,
                        lr=1e-3)
    # Generate a few simple problems
    problems = [sympy.Integral(sympy.diff(x**i, x), x) for i in range(2, 6)]
    stats = trainer.train_iteration(problems)
    assert "policy_loss" in stats
    assert "value_loss" in stats
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rl.py -v`
Expected: FAIL

**Step 3: Implement RL trainer**

```python
# integrate_zero/train/rl.py
import torch
import torch.nn as nn
from integrate_zero.mcts.search import MCTS
from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.data.prefix import sympy_to_prefix

class RLTrainer:
    def __init__(self, model, vocab: Vocabulary,
                 num_candidates: int = 8, search_budget: int = 200,
                 max_steps: int = 20, c_puct: float = 1.5,
                 lr: float = 1e-4):
        self.model = model
        self.vocab = vocab
        self.device = next(model.parameters()).device
        self.mcts = MCTS(model=model, vocab=vocab,
                         max_steps=max_steps, num_candidates=num_candidates,
                         search_budget=search_budget, c_puct=c_puct)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.policy_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.value_criterion = nn.BCELoss()

    def collect_episode(self, problem):
        """Run MCTS on a problem, return trajectory data."""
        self.model.eval()
        trajectory = self.mcts.search(problem)
        solved = trajectory is not None
        return {
            "problem": problem,
            "trajectory": trajectory or [],
            "solved": solved,
        }

    def train_iteration(self, problems: list) -> dict:
        """Run one iteration: collect episodes on all problems, then update."""
        episodes = []
        for problem in problems:
            ep = self.collect_episode(problem)
            episodes.append(ep)

        # Build training batch from episodes
        solve_rate = sum(1 for ep in episodes if ep["solved"]) / max(len(episodes), 1)

        # Train on successful trajectories
        policy_losses = []
        value_losses = []
        self.model.train()

        for ep in episodes:
            reward = 1.0 if ep["solved"] else 0.0
            # Convert trajectory steps to training data
            states = [ep["problem"]] + ep["trajectory"][:-1] if ep["trajectory"] else [ep["problem"]]
            targets = ep["trajectory"] if ep["trajectory"] else []

            for state, target in zip(states, targets):
                loss_dict = self._train_on_pair(state, target, reward)
                policy_losses.append(loss_dict["policy_loss"])
                value_losses.append(loss_dict["value_loss"])

        return {
            "policy_loss": sum(policy_losses) / max(len(policy_losses), 1),
            "value_loss": sum(value_losses) / max(len(value_losses), 1),
            "solve_rate": solve_rate,
        }

    def _train_on_pair(self, state, target, reward: float) -> dict:
        """Train on a single (state, target) pair."""
        try:
            state_tokens = sympy_to_prefix(state)
            target_tokens = sympy_to_prefix(target)
        except Exception:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        # Build input: [BOS] state_tokens [SEP] target_tokens [EOS]
        full = ["BOS"] + state_tokens + ["SEP"] + target_tokens + ["EOS"]
        ids = [self.vocab.token_to_id(t) for t in full if self.vocab.token_to_id(t) is not None]
        if len(ids) < 4:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        sep_id = self.vocab.token_to_id("SEP")
        sep_pos = (input_ids == sep_id).long().argmax(dim=1)

        # Target: only predict after SEP
        target_ids = torch.full_like(input_ids, -100)
        s = sep_pos.item()
        target_ids[0, s+1:] = input_ids[0, s+1:]

        logits, value = self.model(input_ids, sep_pos)

        policy_loss = self.policy_criterion(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            target_ids[:, 1:].reshape(-1)
        )
        value_target = torch.tensor([reward], dtype=torch.float, device=self.device)
        value_loss = self.value_criterion(value, value_target)

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rl.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/train/rl.py tests/test_rl.py
git commit -m "feat: RL training loop with MCTS-based episode collection"
```

---

### Task 13: Evaluation Module

**Files:**
- Create: `integrate_zero/eval/evaluate.py`
- Create: `tests/test_evaluate.py`

**Step 1: Write failing test**

```python
# tests/test_evaluate.py
import sympy
from integrate_zero.eval.evaluate import Evaluator
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary

x = sympy.Symbol("x")

def test_evaluator_runs():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    evaluator = Evaluator(model, vocab)
    problems = [sympy.Integral(x**2, x), sympy.Integral(sympy.sin(x), x)]
    results = evaluator.evaluate(problems, search_budget=10, num_candidates=4)
    assert "solve_rate" in results
    assert "avg_steps" in results
    assert "verification_rate" in results
    assert 0.0 <= results["solve_rate"] <= 1.0

def test_evaluator_sympy_baseline():
    evaluator = Evaluator(model=None, vocab=None)
    problems = [sympy.Integral(x**2, x), sympy.Integral(sympy.sin(x), x)]
    results = evaluator.sympy_baseline(problems)
    assert results["solve_rate"] == 1.0  # SymPy can solve these
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL

**Step 3: Implement evaluator**

```python
# integrate_zero/eval/evaluate.py
import sympy
from integrate_zero.mcts.search import MCTS
from integrate_zero.verify.verifier import is_terminal

class Evaluator:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def evaluate(self, problems: list, search_budget: int = 200,
                 num_candidates: int = 8, max_steps: int = 20) -> dict:
        """Evaluate model on a list of integration problems."""
        mcts = MCTS(model=self.model, vocab=self.vocab,
                     max_steps=max_steps, num_candidates=num_candidates,
                     search_budget=search_budget)

        solved = 0
        total_steps = 0
        total_verified = 0
        total_generated = 0

        for problem in problems:
            trajectory = mcts.search(problem)
            if trajectory is not None:
                solved += 1
                total_steps += len(trajectory)

        return {
            "solve_rate": solved / max(len(problems), 1),
            "avg_steps": total_steps / max(solved, 1),
            "verification_rate": 0.0,  # TODO: track in MCTS
            "total": len(problems),
            "solved": solved,
        }

    def sympy_baseline(self, problems: list) -> dict:
        """Evaluate SymPy's built-in integrate() as baseline."""
        x = sympy.Symbol("x")
        solved = 0
        for problem in problems:
            try:
                result = problem.doit()
                if result is not None and not result.has(sympy.Integral):
                    solved += 1
            except Exception:
                pass
        return {
            "solve_rate": solved / max(len(problems), 1),
            "total": len(problems),
            "solved": solved,
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add integrate_zero/eval/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation module with model and SymPy baseline"
```

---

### Task 14: Main Training Script

**Files:**
- Create: `scripts/train.py`

**Step 1: Create the end-to-end training script**

```python
# scripts/train.py
"""IntegrateZero training script.

Usage:
    python scripts/train.py --phase supervised --num_samples 10000 --epochs 10
    python scripts/train.py --phase rl --checkpoint checkpoints/supervised.pt --iterations 50
    python scripts/train.py --phase eval --checkpoint checkpoints/rl.pt
"""
import argparse
import torch
import sympy
from pathlib import Path

from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.train.supervised import SupervisedTrainer
from integrate_zero.train.rl import RLTrainer
from integrate_zero.eval.evaluate import Evaluator
from integrate_zero.data.generator import generate_training_pair

x = sympy.Symbol("x")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_model(vocab, device):
    return IntegrateZeroModel(
        vocab_size=len(vocab), d_model=384, nhead=6,
        num_layers=8, d_ff=1536,
    ).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["supervised", "rl", "eval"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--problems_per_iter", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrain", type=bool, default=True)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    vocab = Vocabulary()
    model = make_model(vocab, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    Path("checkpoints").mkdir(exist_ok=True)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.phase == "supervised":
        print(f"Generating {args.num_samples} training pairs...")
        dataset = IntegrationDataset(args.num_samples, args.max_depth)
        print(f"Dataset size: {len(dataset)}")

        trainer = SupervisedTrainer(model, dataset, vocab,
                                     batch_size=args.batch_size, lr=args.lr)
        for epoch in range(args.epochs):
            trainer.train_epoch()
            loss = trainer.evaluate_loss()
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")

        trainer.save_checkpoint("checkpoints/supervised.pt")
        print("Saved checkpoint: checkpoints/supervised.pt")

    elif args.phase == "rl":
        trainer = RLTrainer(model, vocab, lr=args.lr)
        for it in range(args.iterations):
            # Generate fresh problems each iteration
            problems = []
            for _ in range(args.problems_per_iter):
                try:
                    f, F = generate_training_pair(args.max_depth)
                    problems.append(sympy.Integral(f, x))
                except Exception:
                    continue

            stats = trainer.train_iteration(problems)
            print(f"Iteration {it+1}/{args.iterations} - "
                  f"Solve rate: {stats['solve_rate']:.3f} - "
                  f"Policy loss: {stats['policy_loss']:.4f} - "
                  f"Value loss: {stats['value_loss']:.4f}")

        torch.save({"model_state_dict": model.state_dict()},
                    "checkpoints/rl.pt")
        print("Saved checkpoint: checkpoints/rl.pt")

    elif args.phase == "eval":
        # Generate test problems
        print("Generating test problems...")
        test_problems = []
        for _ in range(100):
            try:
                f, F = generate_training_pair(args.max_depth)
                test_problems.append(sympy.Integral(f, x))
            except Exception:
                continue

        evaluator = Evaluator(model, vocab)

        print("\n--- Model Evaluation ---")
        model_results = evaluator.evaluate(test_problems)
        print(f"Solve rate: {model_results['solve_rate']:.3f}")
        print(f"Avg steps:  {model_results['avg_steps']:.1f}")

        print("\n--- SymPy Baseline ---")
        sympy_results = evaluator.sympy_baseline(test_problems)
        print(f"Solve rate: {sympy_results['solve_rate']:.3f}")

if __name__ == "__main__":
    main()
```

**Step 2: Verify script runs with --help**

Run: `python scripts/train.py --help`
Expected: shows argument help without errors

**Step 3: Verify supervised phase starts (smoke test)**

Run: `python scripts/train.py --phase supervised --num_samples 100 --epochs 1 --batch_size 8`
Expected: completes one epoch, prints loss, saves checkpoint

**Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat: main training script with supervised, RL, and eval phases"
```

---

## Summary

| Task | Component | Dependencies |
|------|-----------|-------------|
| 1 | Project scaffolding | - |
| 2 | Vocabulary & tokenizer | 1 |
| 3 | Prefix <-> SymPy conversion | 2 |
| 4 | Verifier | 3 |
| 5 | Expression generator | 3 |
| 6 | PyTorch dataset | 2, 3, 5 |
| 7 | Transformer model | 2 |
| 8 | Arity-constrained decoding | 2, 7 |
| 9 | Supervised training | 6, 7 |
| 10 | MCTS | 4 |
| 11 | MCTS-model integration | 3, 7, 8, 10 |
| 12 | RL training | 9, 11 |
| 13 | Evaluation | 11 |
| 14 | Training script | 9, 12, 13 |

Tasks 4 and 5 are independent of each other and can be parallelized. Tasks 6 and 7 are independent and can be parallelized. Tasks 10, 6, 7, 8 can be partially parallelized.
