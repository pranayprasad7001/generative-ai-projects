# Factual Reference: Major Programming Languages

A comparative technical summary of widely adopted programming languages based on design attributes and initial release years.

---

## Language Specifications

| Language | Initial Release | Original Designer | Type System | Primary Paradigm |
| :--- | :--- | :--- | :--- | :--- |
| **Python** | 1991 | Guido van Rossum | Dynamic, Strong | Multi-paradigm (Object-Oriented, Functional) |
| **C++** | 1985 | Bjarne Stroustrup | Static, Strong | Multi-paradigm (Procedural, Object-Oriented) |
| **JavaScript** | 1995 | Brendan Eich | Dynamic, Weak | Event-driven, Prototype-based |
| **Rust** | 2015 | Graydon Hoare | Static, Strong | Systems programming, Concurrent |
| **Go** | 2009 | Robert Griesemer, Rob Pike, Ken Thompson | Static, Strong | Concurrent, Imperative |

---

## Standard Code Examples

### Python (3.x)
```python
def calculate_factorial(n: int) -> int:
    return 1 if n <= 1 else n * calculate_factorial(n - 1)
```

### Rust
```rust
fn calculate_factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n * calculate_factorial(n - 1),
    }
}
```
