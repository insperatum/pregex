import pregex as pre
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch import optim
import string
from collections import Counter

s = "hello"
a = string.ascii_lowercase

t_l = Parameter(torch.zeros(len(a)))
t_p = Parameter(torch.zeros(1))

optimiser = optim.Adam([t_l, t_p])

for i in range(5001):
    optimiser.zero_grad()
    ps = F.softmax(t_l, dim=0)
    p = F.sigmoid(t_p)

    l = pre.CharacterClass(string.ascii_lowercase, name="\\l", normalised_ps=ps)
    r = pre.KleeneStar(l, p=p)
    score = r.match("hello")
    (-score).backward(retain_graph=True)
    optimiser.step()
    if i%1000 == 0:
        print("\nIteration:", i)
        print("p:", p)
        print("Probs:", list(zip(a, ps.tolist())))
        print("Score:", score)

c = Counter(s)
for i,x in enumerate(a):
    assert((ps[i] - (c[x] / len(c))) < 0.01)
assert((p-1/(len(s)+1)).abs() < 0.01)
