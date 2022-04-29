import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from pyruleengine.PyHashcat import py_hashcat

hits = py_hashcat(words_path='/disk/cw/corpora/rockyou-rand3w.txt',
                  rules_path='/disk/cw/downloads/InsidePro-PasswordsPro.rule', 
                  target_path='/disk/cw/corpora/rockyou-tar.txt')

for word, guess, rule, rank, dup in hits:
    print(f"{word}, {guess}, {rule}, {rank}, {dup}")