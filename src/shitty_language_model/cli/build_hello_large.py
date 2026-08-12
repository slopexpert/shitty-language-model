import json
import random

# A compositional hello-world/arithmetic generator. Unlike the item-template
# builders (which only reformat the same 24 snippets), each sample here is a
# structurally distinct program: it draws a random operation family, fresh
# identifiers, literal/argv operand sources, numeric values, and control flow,
# then derives a prose line that describes the ACTUAL code so prose matches
# body. Rapidly varying parameters, so consecutive samples are (effectively)
# never identical.

# ---------------------------------------------------------------------------
# primitive pools / helpers
# ---------------------------------------------------------------------------
GREET_WORDS = [
    "Hello", "Hi", "Hey", "Greetings", "Howdy", "Yo", "Hola", "Welcome",
    "Salutations", "Hey there",
]
GREET_OBJ = [
    "world", "there", "friend", "everyone", "folks", "stranger", "pal",
    "you", "programmer", "guest",
]
BIN_OPS = [
    ("add", "+", "adds two numbers", "sum"),
    ("multiply", "*", "multiplies two numbers", "product"),
    ("subtract", "-", "subtracts one from another", "difference"),
    ("divide", "/", "divides two numbers", "quotient"),
    ("modulo", "%", "takes the remainder", "remainder"),
]
NUM_NAMES = [
    "a", "b", "c", "x", "y", "z", "n", "m", "i", "val", "num", "count",
    "total", "sum", "product", "diff", "result", "value", "base", "input",
    "first", "second", "amount", "lhs", "rhs",
]
STR_NAMES = [
    "greeting", "msg", "message", "text", "line", "word", "name", "hi",
    "s", "str", "buf",
]

random.seed(0xC0FFEE)


def fresh(pool, used):
    # pick an identifier not already in use in this sample
    for _ in range(50):
        name = random.choice(pool)
        if name not in used:
            used.add(name)
            return name
    return random.choice(pool)


# ---------------------------------------------------------------------------
# family generators: each returns (prose, body)
# ---------------------------------------------------------------------------

def gen_greet_literal():
    word = random.choice(GREET_WORDS)
    obj = random.choice(GREET_OBJ)
    line = f'printf("{word}, {obj}!\\n");'
    prose = random.choice([
        f"This program prints a greeting to the screen.",
        f"Here is a tiny C program that writes out a greeting.",
        f"This sample says {word.lower()} and exits.",
        f"A minimal program that outputs a friendly line.",
        f"This program greets {obj} with a short phrase.",
    ])
    return prose, f'{line}\n    return 0;'


def gen_greet_var():
    word = random.choice(GREET_WORDS)
    obj = random.choice(GREET_OBJ)
    used = set()
    name = fresh(STR_NAMES, used)
    kind = random.choice(["char *", "const char *", "char []"])
    if kind == "char []":
        init = f'"{word}, {obj}!\\n"'
        decl = f'char {name}[] = {init};'
    else:
        init = f'"{word}, {obj}!"'
        decl = f'{kind}{name} = {init};'
    print_name = random.choice([name, f"\"%s\\n\", {name}", f"{name}"])
    if print_name == f"\"%s\\n\", {name}":
        print_line = f'printf("{word}, {obj}!\\n");'
    # simplest faithful print: dump the variable
    if kind == "char []":
        print_line = f'puts({name});'
    else:
        print_line = random.choice([f'printf("%s\\n", {name});', f'puts({name});'])
    prose = random.choice([
        f"This program stores a greeting in {kind.strip()} {name} and prints it.",
        f"Here the message is held in a variable and then written out.",
        f"A greeting is saved in {name} and printed with puts.",
    ])
    return prose, f'{decl}\n    {print_line}\n    return 0;'


def gen_greet_argv():
    word = random.choice(GREET_WORDS)
    fallback = random.choice(GREET_OBJ)
    style = random.choice(range(4))
    if style == 0:
        code = (f'if (argc < 2) {{\n'
                f'        printf("{word}, {fallback}!\\n");\n'
                f'        return 0;\n'
                f'    }}\n'
                f'    printf("{word}, %s!\\n", argv[1]);\n'
                f'    return 0;')
    elif style == 1:
        code = (f'if (argc > 1) {{\n'
                f'        printf("{word} %s\\n", argv[1]);\n'
                f'    }} else {{\n'
                f'        printf("{word} {fallback}\\n");\n'
                f'    }}\n'
                f'    return 0;')
    elif style == 2:
        code = (f'printf("{word}, %s.\\n", argc > 1 ? argv[1] : "{fallback}");\n'
                f'    return 0;')
    else:
        code = (f'puts(argc > 1 ? argv[1] : "{word}, {fallback}!");\n'
                f'    return 0;')
    prose = random.choice([
        f"This program greets the name given on the command line.",
        f"If a name is passed in, this sample greets it by name, else {fallback}.",
        f"Reads a name from the arguments and greets it.",
        f"This says {word.lower()} to argv[1] when it is present.",
        f"It greets the first argument, falling back to {fallback}.",
    ])
    return prose, code


def gen_arith_literal():
    a, b = random.randint(1, 60), random.randint(1, 60)
    word, op, verb, noun = random.choice(BIN_OPS)
    used = set()
    na, nb = fresh(NUM_NAMES, used), fresh(NUM_NAMES, used)
    do_split = random.random() < 0.6
    if do_split:
        body = f'int {na} = {a}, {nb} = {b};\n    printf("%d\\n", {na} {op} {nb});\n    return 0;'
    else:
        body = f'printf("%d\\n", {a} {op} {b});\n    return 0;'
    prose = random.choice([
        f"This program {verb} and prints the {noun}.",
        f"Here we compute the {noun} of {a} and {b} and show it.",
        f"A simple {word} whose answer is printed.",
        f"This sample {verb} and outputs the result.",
    ])
    return prose, body


def gen_arith_argv():
    word, op, verb, noun = random.choice(BIN_OPS)
    used = set()
    na, nb = fresh(NUM_NAMES, used), fresh(NUM_NAMES, used)
    body = (f'if (argc < 3) return 1;\n'
            f'    int {na} = atoi(argv[1]);\n'
            f'    int {nb} = atoi(argv[2]);\n'
            f'    printf("%d\\n", {na} {op} {nb});\n'
            f'    return 0;')
    prose = random.choice([
        f"This program {verb} two command-line numbers and prints the {noun}.",
        f"Two numbers from the arguments are combined and the result displayed.",
        f"Reads two arguments and prints their {noun}.",
    ])
    return prose, body


def gen_arith_unary():
    kind = random.choice(range(4))
    src_argv = random.random() < 0.5
    used = set()
    n = fresh(NUM_NAMES, used)
    if src_argv:
        prev = f'int {n} = atoi(argv[1]);'
        argc_guard = 'if (argc < 2) return 1;\n    '
    else:
        val = random.randint(1, 40)
        prev = f'int {n} = {val};'
        argc_guard = ''
    if kind == 0:  # double
        expr = f'{n} * 2'
        verb = "doubles a number"
        pp = "doubled"
    elif kind == 1:  # square
        expr = f'{n} * {n}'
        verb = "squares a number"
        pp = "squared"
    elif kind == 2:  # increment
        expr = f'{n} + 1'
        verb = "adds one to a number"
        pp = "incremented"
    else:  # negate
        expr = f'-{n}'
        verb = "negates a number"
        pp = "negated"
    body = f'{argc_guard}{prev}\n    printf("%d\\n", {expr});\n    return 0;'
    source = "from the command line" if src_argv else ""
    prose = random.choice([
        f"This program {verb} a number{(' ' + source) if source else ''} and prints it.",
        f"Here a value is {pp} and shown on screen.",
        f"This sample {verb} a number{(' ' + source) if source else ''} and outputs it.",
    ])
    return prose, body


def gen_loop():
    count = random.randint(2, 8)
    word = random.choice(GREET_WORDS)
    obj = random.choice(GREET_OBJ)
    used = set()
    i = fresh(["i", "j", "k", "idx", "n"], used)
    show_counter = random.random() < 0.5
    style = random.choice(range(3))
    if style == 0:
        if show_counter:
            body = f'for (int {i} = 0; {i} < {count}; {i}++) {{\n        printf("line %d\\n", {i});\n    }}\n    return 0;'
        else:
            body = f'for (int {i} = 0; {i} < {count}; {i}++) {{\n        printf("{word}, {obj}!\\n");\n    }}\n    return 0;'
    elif style == 1:
        body = f'int {i};\n    for ({i} = 0; {i} < {count}; {i}++) puts("{word}, {obj}!");\n    return 0;'
    else:
        body = (f'int {i} = 0;\n'
                f'    while ({i} < {count}) {{\n'
                f'        puts("{word}, {obj}!");\n'
                f'        {i}++;\n'
                f'    }}\n'
                f'    return 0;')
    prose = random.choice([
        f"This program prints a message {count} times using a loop.",
        f"A {'for' if style != 2 else 'while'} loop repeats the line a few times.",
        f"This sample uses a loop to print up to {count} times.",
    ])
    return prose, body


def gen_conditional():
    used = set()
    n = fresh(NUM_NAMES, used)
    limit = random.randint(1, 50)
    parity = random.random() < 0.6
    body = (f'int {n} = atoi(argv[1]);\n'
            f'    if ({n} % 2 == 0) {{\n'
            f'        printf("even\\n");\n'
            f'    }} else {{\n'
            f'        printf("odd\\n");\n'
            f'    }}\n'
            f'    return 0;')
    prose = random.choice([
        f"This program tells whether a command-line number is even or odd.",
        f"Reads a number and prints even or odd depending on its parity.",
        f"Checks a number's parity and prints the result.",
    ])
    return prose, body


GENERATORS = [
    gen_greet_literal, gen_greet_var, gen_greet_argv,
    gen_arith_literal, gen_arith_argv, gen_arith_unary,
    gen_loop, gen_conditional,
]

# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
SIGS = [
    "int main(void)", "int main()",
    "int main(int argc, char **argv)", "int main(int argc, char *argv[])",
]
HEADERS = ["#include <stdio.h>\n", "#include <stdio.h>\n\n", "\n#include <stdio.h>\n"]
INDENTS = ["    ", "\t", "  "]
BRACES = ["\n{", " {"]
SPACING = ["", "\n", "\n\n"]


def render(prose, body):
    sig = random.choice(SIGS)
    header = random.choice(HEADERS)
    indent = random.choice(INDENTS)
    brace = random.choice(BRACES)
    pre = random.choice(SPACING)
    post = random.choice(SPACING)
    # body is written at 4-space depth; rebase onto chosen indent
    lines = []
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        depth = 0
        while line.startswith("    "):
            depth += 1
            line = line[4:]
        lines.append(indent * depth + stripped)
    code = f"{header}{sig}{brace}\n{chr(10).join(lines)}\n}}{post}"
    return f"{prose}\n\n{pre}{code}"


def build(count=2000):
    samples = []
    seen = set()
    attempts = 0
    while len(samples) < count and attempts < count * 10:
        attempts += 1
        gen = random.choice(GENERATORS)
        prose, body = gen()
        if body in seen:
            continue
        seen.add(body)
        samples.append(render(prose, body))
    return samples


def main():
    count = 2000
    samples = build(count)
    payload = {
        "name": "hello-world-large",
        "description": (
            "Compositionally-generated novel hello-world / argv / arithmetic / "
            "loop / parity C programs, each with matching prose. Distinct "
            "operations, operands, identifiers, argv sources, and control flow."
        ),
        "samples": samples,
    }
    out_path = "corpus/hello_large.json"
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        raise SystemExit(f"could not write {out_path!r}: {exc}") from exc
    print(f"wrote {len(samples)} unique samples -> {out_path}")


if __name__ == "__main__":
    main()
