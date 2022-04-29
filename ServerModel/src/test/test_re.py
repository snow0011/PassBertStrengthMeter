import re


def check(l):
    return re.findall(u'.*?[\n。]+', l)


def wrapper():
    print(check("hello。world。\n"))
    pass


if __name__ == '__main__':
    wrapper()
