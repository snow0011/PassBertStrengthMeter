from collections import defaultdict
from typing import Dict, List, Generator, Tuple


def read_rules(rule_path) -> List[str]:
    with open(rule_path, 'r') as f_rule:
        raw_rules = [r.strip('\r\n') for r in f_rule]
        rules = [r for r in raw_rules if r and r[0] != '#']
    return rules


def read_words(words_path: str, start_at: int = 1) -> Generator[Tuple[int, str], None, None]:
    """
    start_at specified line number.
    """
    with open(words_path, 'r') as f_words:
        start_at = max(1, start_at)
        for _ in range(1, start_at):
            f_words.readline()
        idx = start_at - 1
        for line in f_words:
            line = line.strip('\r\n')
            yield idx, line
            idx += 1


def read_target(target_path) -> Dict[str, int]:
    pwd_set = defaultdict(int)
    with open(target_path, 'r') as f_target:
        for line in f_target:
            line = line.strip('\r\n')
            pwd_set[line] += 1
        pass
    return pwd_set
