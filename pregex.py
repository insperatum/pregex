#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from scipy.stats import geom
from collections import namedtuple, Counter
try: from queue import PriorityQueue
except ImportError: from Queue import PriorityQueue

import random
import math
from string import ascii_letters, digits, ascii_lowercase, ascii_uppercase, whitespace, printable
printable = printable[:-4]
whitespace = [x for x in whitespace if x in printable] 

import numpy as np

PartialMatch = namedtuple("PartialMatch", ["numCharacters", "score", "reported_score", "continuation", "state"])

OPEN = "BRACKET_OPEN"
CLOSE = "BRACKET_CLOSE"

class Pregex(namedtuple("Pregex", ["type", "arg"])):
    def __new__(cls, arg):
        return super(Pregex, cls).__new__(cls, cls.__name__, arg)

    def __getnewargs__(self):
        return (self.arg,)

    def __repr__(self):
        return str("(" + type(self).__name__ + " " + repr(self.arg) + ")")

    def str(self, f=str, escape_strings=True):
        char_map = {
            dot: ".",
            d: "\\d",
            s: "\\s",
            w: "\\w",
            l: "\\l",
            u: "\\u",
            KleeneStar: "*",
            Plus: "+",
            Maybe: "?",
            Alt: "|",
            OPEN: "(",
            CLOSE: ")"
        }
        flat = flatten(self, char_map=char_map, escape_strings=escape_strings)
        return "".join([x if type(x) is str else repr(x) if issubclass(type(x), Pregex) else f(x) for x in flat])

    def __str__(self):
        return self.str()

    def flatten(self, char_map={}, escape_strings=False):
        return [char_map.get(type(self), self)]

    def sample(self, state=None):
        """
        Returns a sample
        """
        raise NotImplementedError()

    def consume(self, s, state=None):
        """
        :param s str:
        Consume some of s
        Yield the score, the number of tokens consumed, the remainder of the regex, and the final state
        Returns generator(PartialMatch)
        """
        raise NotImplementedError()

    def leafNodes(self):
        """
        returns a list of leaves for this regex
        """
        return []

# Viterbi-style
#
#    def match(self, string, state=None, mergeState=True, returnPartials=False):
#        """
#        :param bool mergeState: if True, only retain the highest scoring state for each continuation 
#        """
#        initialState = state
#        partialsAt = [[] for i in range(len(string)+1)]
#        finalMatches = [[] for i in range(len(string)+1)]
#        partialsAt[0] = [(0, self, initialState, 0)]
#        # partialsAt[num characters consumed] = [(score, continuation, state, reported_score), ...]
#        
#        def merge(partials):
#            #partials: [(score, continuation, state), ...]
#            best = {} # best: continuation -> (score, continuation, state, reported_score)
#            for x in partials:
#                key = x[1] if mergeState else x[1:]
#                x_best = best.get(key, None)
#                if x_best is None or x_best[0] < x[0]:
#                    best[key] = x
#            return list(best.values())
#        
#        for i in range(len(string)+1):
#            #Merge to find MAP
#            partialsAt[i] = merge(partialsAt[i])
#            #Match some characters
#            remainder = string[i:]
#            while partialsAt[i]:
#                score, continuation, state, reported_score = partialsAt[i].pop()
#                if continuation is None:
#                    finalMatches[i].append((score, continuation, state, reported_score))
#                    continue
#                for remainderMatch in continuation.consume(remainder, state):
#                    j = i + remainderMatch.numCharacters
#                    if i==j and continuation == remainderMatch.continuation and state == remainderMatch.state:
#                        raise Exception()
#                    else:
#                        partialsAt[j].append((score + remainderMatch.score, remainderMatch.continuation, remainderMatch.state, reported_score + remainderMatch.reported_score))
#
#        def getOutput(matches):
#            matches = merge(matches)
#            if matches:
#                score, _, state, reported_score = matches[0]
#            else:
#                state = None
#                reported_score = float("-inf")
#
#            if initialState is None:
#                return reported_score
#            else:
#                return reported_score, state
#
#        if returnPartials:
#            return [(numCharacters, getOutput(finalMatches[numCharacters])) for numCharacters in range(len(finalMatches)) if finalMatches[numCharacters]]
#        else:
#            return getOutput(finalMatches[-1])
#

# Dijkstra-style

    def match(self, string, state=None, mergeState=True, returnPartials=False):
        """
        :param bool mergeState: if True, only retain the highest scoring state for each continuation 
        """
        initialState = state
        partialsAt = [[] for i in range(len(string)+1)]
        finalMatches = [[] for i in range(len(string)+1)]
        Node = namedtuple("Node", ("numCharacters", "continuation", "state"))
        start = PartialMatch(0, 0, 0, self, state)
        visited = {Node(0, self, state)} 
        queue = PriorityQueue()
        queue.put((0, start)) #Priority 0
        
        solution = None
        while not queue.empty() and solution is None:
            priority, current = queue.get()
            remainder = string[current.numCharacters:]
            for remainderMatch in current.continuation.consume(remainder, current.state):
                assert not (remainderMatch.numCharacters==0 and current.continuation==remainderMatch.continuation and current.state==remainderMatch.state)
                numCharacters = current.numCharacters + remainderMatch.numCharacters
                newNode = Node(numCharacters, remainderMatch.continuation, remainderMatch.state)
                if newNode not in visited:
                    visited.add(newNode)
                    newMatch = PartialMatch(
                        numCharacters,
                        current.score + remainderMatch.score,
                        current.reported_score + remainderMatch.reported_score,
                        remainderMatch.continuation,
                        remainderMatch.state)
                    if newMatch.continuation is None:
                        if newMatch.numCharacters == len(string):
                            solution = newMatch
                            break
                    else:
                        queue.put((-remainderMatch.score, newMatch))

        def getOutput(match):
            if match is not None:
                state = match.state
                reported_score = match.reported_score
            else:
                state = None
                reported_score = float("-inf")

            if initialState is None:
                return reported_score
            else:
                return reported_score, state

        if returnPartials:
            raise NotImplementedError
#            return [(numCharacters, getOutput(finalMatches[numCharacters])) for numCharacters in range(len(finalMatches)) if finalMatches[numCharacters]]
        else:
            return getOutput(solution)

class CharacterClass(Pregex):
    def __new__(cls, values, ps=None, name=None, normalised_ps=None):
        if normalised_ps is None:
            if ps is None:
                ps = [1/len(values) for value in values]
            else:
                #do normalization
                assert len(ps) == len(values)
                ps = [p/sum(ps) for p in ps]
        else:
            ps = normalised_ps
        return super(CharacterClass, cls).__new__(cls, (tuple(values), tuple(ps), name))

    def __getnewargs__(self):
        return (self.values, None, self.name, self.ps)

    @property
    def values(self):
        return self.arg[0]

    @property
    def ps(self):
        return self.arg[1]

    @property
    def name(self):
        return self.arg[2]

    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            return "[" + "".join(self.values) + "]"

    def flatten(self, char_map={}, escape_strings=False):
        return [char_map.get(self, self)]

    def sample(self, state=None):
        return np.random.choice(self.values, p=self.ps)

    def consume(self, s, state=None):
        if len(s)>=1 and s[:1] in self.values:
            score = math.log(self.ps[self.values.index(s[:1])])
            yield PartialMatch(numCharacters=1, score=score, reported_score=score, continuation=None, state=state)

#Uniform Frequences
dot = CharacterClass(printable, name=".") #Don't match newline characters
d = CharacterClass(digits, name="\\d")
s = CharacterClass(whitespace, name="\\s")
w = CharacterClass(ascii_letters + digits, name="\\w")
l = CharacterClass(ascii_lowercase, name="\\l")
u = CharacterClass(ascii_uppercase, name="\\u")

#Empirical Frequencies from Reuters articles https://trec.nist.gov/data/reuters/reuters.html
_emp = Counter({' ': 2643759, 'e': 1386846, 't': 1012840, 'a': 966445, 'n': 868336, 'i': 864695, 'o': 854979, 'r': 827118, 's': 818629, 'l': 507744, 'd': 503130, 'h': 413245, 'c': 394475, '\n': 328005, 'u': 317870, 'm': 286203, 'p': 273927, 'f': 248498, 'g': 202023, '.': 174421, 'y': 171873, 'b': 163239, 'w': 150923, ',': 136229, 'v': 123365, '0': 93385, 'k': 78030, '1': 77905, 'T': 52995, 'S': 48747, 'C': 47133, '2': 43486, 'R': 40720, '8': 40276, '5': 39868, 'A': 39088, '9': 38532, 'x': 36389, '3': 34498, 'I': 32991, '-': 32765, '6': 30291, 'E': 29527, '4': 29007, '7': 28783, 'M': 28612, 'B': 27432, '"': 24856, "'": 23790, 'P': 21916, 'U': 21007, 'N': 20130, 'F': 19253, 'D': 17241, 'q': 15912, 'L': 15848, 'G': 14689, 'J': 13907, 'H': 13809, 'O': 12954, 'W': 12698, 'j': 9744, 'z': 9092, '/': 8198, '<': 6959, '>': 6949, 'K': 6008, ')': 5229, '(': 5219, 'V': 4033, 'Y': 3975, ':': 1899, 'Q': 1578, 'Z': 1360, 'X': 1037, ';': 117, '?': 73, '\x7f': 49, '^': 35, '&': 32, '+': 24, '[': 11, ']': 10, '$': 8, '!': 8, '*': 7, '=': 4, '~': 3, '_': 2, '\t': 2, '@': 1, '\x1b': 1, '{': 1, '\xfc': 1, '\x1e': 1, '\x05': 1})
def _natural_probs(chars):
    total = sum(_emp[x] for x in chars)
    return [_emp[x]/total for x in chars]
dot_natural = CharacterClass(printable, name=".", ps=_natural_probs(printable))
d_natural = CharacterClass(digits, name="\\d", ps=_natural_probs(digits))
s_natural = CharacterClass(whitespace, name="\\s", ps=_natural_probs(whitespace))
w_natural = CharacterClass(ascii_letters + digits, name="\\w", ps=_natural_probs(ascii_letters+digits))
l_natural = CharacterClass(ascii_lowercase, name="\\l", ps=_natural_probs(ascii_lowercase))
u_natural = CharacterClass(ascii_uppercase, name="\\u", ps=_natural_probs(ascii_uppercase))

class Named(Pregex):
    def __new__(cls, name, value): return super(Named, cls).__new__(cls, (name, value))
    def sample(self, state=None): return self.arg[1].sample(state)
    def consume(self, s, state=None): return self.arg[1].consume(s, state)
    def leafNodes(self): return self.arg[1].leafNodes()
    def flatten(self, char_map={}, escape_strings=False): return [self.arg[0]]

class String(Pregex):
    def flatten(self, char_map={}, escape_strings=False):
        if escape_strings:
            return list(self.arg.replace("\\", "\\\\").replace(".", "\\.").replace("+", "\\+").replace("*", "\\*").replace("?", "\\?").replace("|", "\\|").replace("(", "\\(").replace(")", "\\)"))
        else:
            return list(self.arg)
        
    def sample(self, state=None):
        return self.arg

    def consume(self, s, state=None):
        if s[:len(self.arg)]==self.arg:
            yield PartialMatch(numCharacters=len(self.arg), score=0, reported_score=0, continuation=None, state=state)

class Concat(Pregex):
    def __new__(cls, values):
        return super(Concat, cls).__new__(cls, tuple(values))

    def __getnewargs__(self):
        return (self.values,)

    @property
    def values(self):
        return self.arg

    def flatten(self, char_map={}, escape_strings=False):
        return sum([flatten(x, char_map, escape_strings) for x in self.values], [])

    def sample(self, state=None):
        return "".join(value.sample(state) for value in self.values)

    def leafNodes(self):
        return [x for child in self.values for x in child.leafNodes()]

    def consume(self, s, state=None):
        for partialMatch in self.values[0].consume(s, state):
            if partialMatch.continuation is None:
                if len(self.values)==1:
                    continuation = None
                elif len(self.values)==2:
                    continuation = self.values[1]
                else:
                    continuation = Concat(self.values[1:])
            else:
                if len(self.values)==1:
                    continuation = partialMatch.continuation
                else:
                    continuation = Concat((partialMatch.continuation,) + self.values[1:])
            yield partialMatch._replace(continuation=continuation)

class Alt(Pregex):
    def __new__(cls, values, ps=None):
        if ps is None:
            ps = (1/len(values),) * len(values)
        return super(Alt, cls).__new__(cls, (tuple(ps), tuple(values)))

    def __getnewargs__(self):
        return (self.values, self.ps)

    @property
    def ps(self):
        return self.arg[0]

    @property
    def values(self):
        return self.arg[1]

    def flatten(self, char_map={}, escape_strings=False):
        def bracket(value):
            if (type(value) is String and len(value.arg)>1) or type(value) == Concat:
                return [char_map.get(OPEN, OPEN)] + flatten(value, char_map, escape_strings) + [char_map.get(CLOSE, CLOSE)]
            else:
                return flatten(value, char_map, escape_strings)
        
        out = []
        for i in range(len(self.values)):
            if i>0: out.append(char_map.get(type(self), type(self)))
            out.extend(bracket(self.values[i]))
        return out

    def sample(self, state=None):
        values = np.empty((len(self.values,)), dtype=object)
        for i in range(len(self.values)): values[i] = self.values[i] #Required so that numpy doesn't make a 2d array (because namedtuple is iterable :/ ...)

        value = np.random.choice(values, p=self.ps)
        return value.sample(state)

    def leafNodes(self):
        return [x for child in self.values for x in child.leafNodes()]

    def consume(self, s, state=None):
        for p, value in zip(self.ps, self.values):
            for partialMatch in value.consume(s, state):
                extraScore = math.log(p)
                yield partialMatch._replace(score=partialMatch.score+extraScore, reported_score=partialMatch.reported_score+extraScore)

class NonEmpty(Pregex):
    """
    (Used in KleeneStar.match)
    """
    def consume(self, s, state=None):
        stack = [PartialMatch(numCharacters=0, score=0, reported_score=0, continuation=self.arg, state=state)]
        while stack:
            p = stack.pop()
            if p.continuation is not None:
                for p2 in p.continuation.consume(s[p.numCharacters:], p.state):
                    partialMatch = p2._replace(score=p.score+p2.score, reported_score=p.reported_score+p2.reported_score)
                    if partialMatch.numCharacters>0:
                        yield partialMatch
                    else:
                        stack.append(partialMatch)

class KleeneStar(Pregex):
    def __new__(cls, arg, p=0.5):
        return super(KleeneStar, cls).__new__(cls, (p, arg))

    def __getnewargs__(self):
        return self.val, self.p

    @property
    def p(self):
        return self.arg[0]

    @property
    def val(self):
        return self.arg[1]

    def __repr__(self):
        return str("(" + type(self).__name__ + " " + str(self.p) + " " + repr(self.val) + ")")

    def flatten(self, char_map={}, escape_strings=False):
        if type(self.val) in (Alt, Concat) or (type(self.val)==String and len(self.val.arg)>1):
            return [char_map.get(OPEN, OPEN)] + flatten(self.val, char_map, escape_strings) + [char_map.get(CLOSE, CLOSE), char_map.get(type(self), type(self))]
        else:
            return flatten(self.val, char_map, escape_strings) + [char_map.get(type(self), type(self))]
            
    def sample(self, state=None):
        n = geom.rvs(self.p, loc=-1)
        return "".join(self.val.sample(state) for i in range(n))

    def leafNodes(self):
        return self.val.leafNodes()

    def consume(self, s, state=None):
        yield PartialMatch(score=math.log(self.p), reported_score=math.log(self.p), numCharacters=0, continuation=None, state=state)
        for partialMatch in NonEmpty(self.val).consume(s, state):
            assert(partialMatch.numCharacters > 0)
            # Force matching to be nonempty, to avoid infinite recursion when matching fo?* -> foo
            # This is only valid for MAP. If we want to get the marginal, we should first calculate 
            # probability q=P(o?->ε), then multiply all partialmatches by 1/[1-q(1-p))]. (TODO)

            if partialMatch.continuation is None:
                continuation = KleeneStar(self.val)
            else:
                continuation = Concat((partialMatch.continuation, KleeneStar(self.val)))

            extraScore = math.log(1-self.p) 
            yield partialMatch._replace(score=partialMatch.score+extraScore, reported_score=partialMatch.reported_score+extraScore, continuation=continuation)


class Plus(Pregex):
    def __new__(cls, arg, p=0.5):
        return super(Plus, cls).__new__(cls, (p, arg))

    def __getnewargs__(self):
        return self.arg[1], self.arg[0]

    @property
    def p(self):
        return self.arg[0]

    @property
    def val(self):
        return self.arg[1]

    def __repr__(self):
        return str("(" + type(self).__name__ + " " + str(self.p) + " " + repr(self.val) + ")")

    def flatten(self, char_map={}, escape_strings=False):
        if type(self.val) in (Alt, Concat) or (type(self.val)==String and len(self.val.arg)>1):
            return [char_map.get(OPEN, OPEN)] + flatten(self.val, char_map, escape_strings) + [char_map.get(CLOSE, CLOSE), char_map.get(type(self), type(self))]
        else:
            return flatten(self.val, char_map, escape_strings) + [char_map.get(type(self), type(self))]

    def sample(self, state=None):
        n = geom.rvs(self.p, loc=0)
        return "".join(self.val.sample(state) for i in range(n))    

    def leafNodes(self):
        return self.val.leafNodes()

    def consume(self, s, state=None):
        for partialMatch in self.val.consume(s, state):
            if partialMatch.continuation is None:
                continuation = KleeneStar(self.val)
            else:
                continuation = Concat((partialMatch.continuation, KleeneStar(self.val)))

            yield partialMatch._replace(continuation=continuation)    


class Maybe(Pregex):
    def __new__(cls, arg, p=0.5):
        return super(Maybe, cls).__new__(cls, (p, arg))

    def __getnewargs__(self):
        return self.arg[1], self.arg[0]

    @property
    def p(self):
        return self.arg[0]

    @property
    def val(self):
        return self.arg[1]

    def __repr__(self):
        return str("(" + type(self).__name__ + " " + str(self.p) + " " + repr(self.val) + ")")

    def flatten(self, char_map={}, escape_strings=False):
        if type(self.val) in (Alt, Concat) or (type(self.val)==String and len(self.val.arg)>1):
            return [char_map.get(OPEN, OPEN)] + flatten(self.val, char_map, escape_strings) + [char_map.get(CLOSE, CLOSE), char_map.get(type(self), type(self))]
        else:
            return flatten(self.val, char_map, escape_strings) + [char_map.get(type(self), type(self))]

    def sample(self, state=None):
        if random.random() < self.p:
            return self.val.sample(state)
        else:
            return ""

    def leafNodes(self):
        return self.val.leafNodes()

    def consume(self, s, state=None):
        yield PartialMatch(score=math.log(1-self.p), reported_score=math.log(1-self.p), numCharacters=0, continuation=None, state=state)
        for partialMatch in self.val.consume(s, state):
            extraScore = math.log(self.p)
            yield partialMatch._replace(score=partialMatch.score+extraScore, reported_score=partialMatch.reported_score+extraScore)


# ------------------------------------

def flatten(obj, char_map, escape_strings):
    if issubclass(type(obj), Pregex):
        return obj.flatten(char_map, escape_strings)
    else:
        return [obj]

# ------------------------------------

class ParseException(Exception):
    pass

def create(seq, lookup=None, natural_frequencies=False):
    """
    Seq is a string or a list
    """
    def head(x):
        if type(seq) is str:
            return {"*":KleeneStar, "+":Plus, "?":Maybe, "|":Alt, "(":OPEN, ")":CLOSE}.get(x[0], x[0])
        elif type(seq) is list:
            return x[0]

    def precedence(x):
        return {KleeneStar:2, Plus:2, Maybe:2, Alt:1, OPEN:0, CLOSE:-1}.get(x, 0)

    def parseToken(seq):
        if len(seq) == 0: raise ParseException()

        if lookup is not None and seq[0] in lookup:
            return lookup[seq[0]], seq[1:]

        elif issubclass(type(seq[0]), Pregex):
            return seq[0], seq[1:]

        elif type(seq) is str and (seq[:2] in ("\\*", "\\+", "\\?", "\\|", "\\(", "\\)", "\\.", "\\\\", "\\d", "\\s", "\\w", "\\l", "\\u") or seq[:1] == "."):
            if   seq[:2] == "\\*": return String("*"), seq[2:]
            elif seq[:2] == "\\+": return String("+"), seq[2:]
            elif seq[:2] == "\\?": return String("?"), seq[2:]
            elif seq[:2] == "\\|": return String("|"), seq[2:]
            elif seq[:2] == "\\(": return String("("), seq[2:]
            elif seq[:2] == "\\)": return String(")"), seq[2:]
            elif seq[:2] == "\\.": return String("."), seq[2:]
            elif seq[:2] == "\\\\": return String("\\"), seq[2:]

            elif seq[:2] == "\\d": return d_natural if natural_frequencies else d, seq[2:]
            elif seq[:2] == "\\s": return s_natural if natural_frequencies else s, seq[2:]
            elif seq[:2] == "\\w": return w_natural if natural_frequencies else w, seq[2:]
            elif seq[:2] == "\\l": return l_natural if natural_frequencies else l, seq[2:]
            elif seq[:2] == "\\u": return u_natural if natural_frequencies else u, seq[2:]
            elif seq[:1] == ".":  return dot_natural if natural_frequencies else dot, seq[1:]

        elif head(seq) == OPEN:
                if len(seq)<=1: raise ParseException() #Lookahead
                inner_lhs, inner_remainder = parseToken(seq[1:])
                rhs, seq = parse(inner_lhs, inner_remainder, -1, True)
                return rhs, seq[1:]

        elif type(seq[0]) is str and seq[0] in printable:
            return String(seq[0]), seq[1:]

        else:
            raise ParseException()

    def parse(lhs, remainder, min_precedence=0, inside_brackets=False):
        if not remainder:
            if inside_brackets: raise ParseException()
            return lhs, remainder

        else:
            if precedence(head(remainder)) < min_precedence:
                return lhs, remainder
            
            elif head(remainder) == CLOSE:
                if not inside_brackets: raise ParseException()
                return lhs, remainder

            elif head(remainder) not in (KleeneStar, Plus, Maybe, Alt): #Atom
                rhs, remainder = parseToken(remainder)

                while remainder and head(remainder) != CLOSE:
                    rhs, remainder = parse(rhs, remainder, 0, inside_brackets)

                if type(lhs) is String and type(rhs) is String:
                    return String(lhs.arg + rhs.arg), remainder
                elif type(lhs) is String and type(rhs) is Concat and type(rhs.values[0]) is String:
                    return Concat((String(lhs.arg + rhs.values[0].arg),) + rhs.values[1:]), remainder
                elif type(rhs) is Concat:
                    return Concat((lhs,) + rhs.values), remainder
                else:
                    return Concat([lhs, rhs]), remainder

            else:
                op, remainder = head(remainder), remainder[1:]
                if op in (KleeneStar, Plus, Maybe): 
                    #Don't need to look right
                    lhs = op(lhs)
                    return parse(lhs, remainder, min_precedence, inside_brackets)
                elif op == Alt:
                    #Need to look right
                    rhs, remainder = parseToken(remainder)

                    while remainder and precedence(head(remainder)) >= precedence(op):
                        rhs, remainder = parse(rhs, remainder, precedence(op), inside_brackets)

                    if type(rhs) is Alt:
                        lhs = Alt((lhs,) + rhs.values)
                    else:
                        lhs = Alt([lhs, rhs])
                    return parse(lhs, remainder, min_precedence, inside_brackets)

    lhs, remainder = parseToken(seq)
    return parse(lhs, remainder)[0]






# ------------------------------------------------------------------------------------------------
class Wrapper(Pregex):
    """
    :param state->value arg.sample: 
    :param string,state->score,state arg.match:
    """
    def __repr__(self):
        return "(" + type(self.arg).__name__ + ")"

    def sample(self, state=None):
        return self.arg.sample(state)

    def consume(self, s, state=None):
        for i in range(len(s)+1):
            matchScore, newState = self.arg.match(s[:i], state)
            if matchScore > float("-inf"):
                yield PartialMatch(numCharacters=i, score=matchScore, reported_score=matchScore, continuation=None, state=newState)


