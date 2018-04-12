# pregex
Probabilistic regular expressions

- Kleene star (and +) use geometric distributions on length. 
- Currently, score returns likelihood of *most probable* execution trace, rather than marginalising.
- Add new primitives with Wrapper class
- Primitives may be stateful (so log p(/AA/->xx) is not necessarily 2 * log p(/A/->x))

Usage:

```
import pregex as pre
r = pre.create("\\d+\\l+\\u+\\s")
samp = r.sample() //'3gclxbZ\t'
score = r.match("123abcABC ") //-34.486418601378595
```

# Todo:
- [ ] Add marginal likelihood (subtlety to avoid infinite recursion in KleeneStar.consume)