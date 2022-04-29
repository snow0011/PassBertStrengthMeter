import random
import string
import time

import PyRuleEngine


def main():
    input_list = [''.join(random.choice(string.printable) for _ in range(10))
                  for _ in range(1000000)]
    engines = []
    for rule in [':', 'l', 'u', 'c', 'C', 't', 'T3', 'r', 'd', 'p2', 'f',
                 '{', '}', '$1', '^1', '[', ']', 'D3', 'x04', 'O12', 'i4!',
                 'o3$', "'6", 'ss$', '@s', 'z2', 'Z2', 'q', 'lMX428', 'uMl4',
                 'rMr6', 'lMuX084']:
        engines.append((rule, PyRuleEngine.RuleEngine([rule])))
    for rule, engine in engines:
        start = time.time()
        for base in input_list:
            list(engine.apply(base))
        print('%s%s%s' % (rule, ' ' * (10 - len(rule)),
                          round(time.time() - start, 4)))


if __name__ == '__main__':
    main()
