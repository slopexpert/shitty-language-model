import json
import random

# Deliberately varied hello-world / simple-math C programs, each preceded by a
# short prose explanation that actually matches the code. The prose teaches the
# model natural language that introduces code; the code teaches simple C.
# Widely varied style (naming, formatting, brace placement, signatures) so
# training never latches onto one exact template.

# Each item pairs a code body with prose lines that describe it.
# Bodies are written with leading 4-space indentation (re-base-able later).
ITEMS = [
    # plain hello world
    {
        "prose": [
            "This program prints a greeting to the screen.",
            "A tiny hello-world example written in C.",
            "This sample says hello and then exits.",
            "Here is a minimal program that writes some text.",
            "This program writes a short message and returns.",
        ],
        "code": 'printf("Hello, world!\\n");\n    return 0;',
    },
    {
        "prose": [
            "This program prints a greeting to the screen.",
            "Here is the simplest C program that outputs some text.",
            "This sample prints a short message and exits cleanly.",
        ],
        "code": 'printf("hello world\\n");\n    return 0;',
    },
    {
        "prose": [
            "This program greets the user with a short phrase.",
            "A minimal program that prints a friendly line.",
        ],
        "code": 'printf("Hi there\\n");\n    return 0;',
    },
    {
        "prose": [
            "This program prints a greeting using puts.",
            "This sample writes a single line of text.",
        ],
        "code": 'puts("Hello, world!");\n    return 0;',
    },
    # greeting via a variable
    {
        "prose": [
            "This program stores a greeting in a variable and prints it.",
            "Here a message is saved in a variable and then written out.",
        ],
        "code": 'char *greeting = "Hello, world!";\n    printf("%s\\n", greeting);\n    return 0;',
    },
    {
        "prose": [
            "This sample keeps the message in a constant and prints it.",
            "The greeting is held in a const string and written with puts.",
        ],
        "code": 'const char *msg = "hello there";\n    puts(msg);\n    return 0;',
    },
    {
        "prose": [
            "Here a character array holds the message that gets printed.",
            "This program stores text in an array and prints the array.",
        ],
        "code": 'char message[] = "Good morning!\\n";\n    printf(message);\n    return 0;',
    },
    # greet the name from the command line
    {
        "prose": [
            "This program greets the user by the name given on the command line.",
            "If a name is passed in, this sample says hello to it by name.",
            "This example reads a name from the arguments and greets it.",
        ],
        "code": 'if (argc < 2) {\n        printf("Hello, stranger!\\n");\n        return 0;\n    }\n    printf("Hello, %s!\\n", argv[1]);\n    return 0;',
    },
    {
        "prose": [
            "This program says hello to argv[1] when it is present.",
            "It greets the first command-line argument by name.",
        ],
        "code": 'if (argc > 1) {\n        printf("Hello %s\\n", argv[1]);\n    } else {\n        printf("Hello nobody\\n");\n    }\n    return 0;',
    },
    {
        "prose": [
            "This sample greets the given name, or a default if none was provided.",
            "A short program that greets a name, falling back to a default.",
        ],
        "code": 'printf("Hi, %s.\\n", argc > 1 ? argv[1] : "guest");\n    return 0;',
    },
    {
        "prose": [
            "This program prints the first argument if one was supplied.",
            "It outputs the argument from the command line when available.",
        ],
        "code": 'puts(argc > 1 ? argv[1] : "Hello, world!");\n    return 0;',
    },
    # simple arithmetic on literals
    {
        "prose": [
            "This program adds two numbers together and prints the sum.",
            "Here we compute a small sum and show the result.",
        ],
        "code": 'int a = 4, b = 7;\n    printf("sum = %d\\n", a + b);\n    return 0;',
    },
    {
        "prose": [
            "This program multiplies two numbers and prints the product.",
            "A simple multiplication whose answer is printed.",
        ],
        "code": 'int x = 21, y = 21;\n    printf("%d\\n", x * y);\n    return 0;',
    },
    {
        "prose": [
            "This sample doubles a number and prints it.",
            "Here a value is multiplied by two and shown on screen.",
        ],
        "code": 'int n = 12;\n    printf("twice = %d\\n", n * 2);\n    return 0;',
    },
    {
        "prose": [
            "This program subtracts one number from another and prints it.",
            "A short example that prints the result of a subtraction.",
        ],
        "code": 'int p = 10, q = 3;\n    printf("%d\\n", p - q);\n    return 0;',
    },
    {
        "prose": [
            "This program divides two numbers and prints the quotient.",
            "Here we divide two values and output the result.",
        ],
        "code": 'int a = 40, b = 5;\n    printf("quotient %d\\n", a / b);\n    return 0;',
    },
    # arithmetic read from the command line
    {
        "prose": [
            "This program adds two integers read from the command line.",
            "It takes two arguments, adds them, and prints the sum.",
        ],
        "code": 'if (argc < 3) return 1;\n    int a = atoi(argv[1]);\n    int b = atoi(argv[2]);\n    printf("%d\\n", a + b);\n    return 0;',
    },
    {
        "prose": [
            "This program reads two numbers and prints their product.",
            "Two command-line numbers are multiplied and displayed.",
        ],
        "code": 'if (argc != 3) return 1;\n    int x = atoi(argv[1]);\n    int y = atoi(argv[2]);\n    printf("product = %d\\n", x * y);\n    return 0;',
    },
    {
        "prose": [
            "This program squares the number given on the command line.",
            "It reads one argument and prints its square.",
        ],
        "code": 'if (argc < 2) return 1;\n    int v = atoi(argv[1]);\n    printf("square is %d\\n", v * v);\n    return 0;',
    },
    {
        "prose": [
            "This program adds one to the number passed on the command line.",
            "It takes an argument and prints that value incremented by one.",
        ],
        "code": 'if (argc < 2) return 1;\n    int base = atoi(argv[1]);\n    printf("%d\\n", base + 1);\n    return 0;',
    },
    # loop that prints several times
    {
        "prose": [
            "This program prints a greeting three times using a loop.",
            "A for loop repeats the message a few times.",
        ],
        "code": 'for (int i = 0; i < 3; i++) {\n        printf("hello\\n");\n    }\n    return 0;',
    },
    {
        "prose": [
            "This sample uses a loop to say hi five times.",
            "A short loop prints a line multiple times.",
        ],
        "code": 'int i;\n    for (i = 0; i < 5; i++) puts("hi");\n    return 0;',
    },
    {
        "prose": [
            "Here a loop prints a numbered line each iteration.",
            "This program shows the loop counter while it runs.",
        ],
        "code": 'for (int i = 0; i < 3; i++) {\n        printf("line %d\\n", i);\n    }\n    return 0;',
    },
    {
        "prose": [
            "This program uses a while loop to print a greeting.",
            "A while loop repeats the message until a counter runs out.",
        ],
        "code": 'int i = 0;\n    while (i < 3) {\n        puts("Hello");\n        i++;\n    }\n    return 0;',
    },
]

# CLI-able main signatures to rotate between
SIGS = [
    "int main(void)",
    "int main()",
    "int main(int argc, char **argv)",
    "int main(int argc, char *argv[])",
    "int main(int argc, char **argv)",
]

HEADERS = [
    "#include <stdio.h>\n",
    "#include <stdio.h>\n\n",
    "\n#include <stdio.h>\n",
]
INDENTS = ["    ", "\t", "  "]
OPEN_BRACE_STYLES = ["\n{", " {"]
SPACING = ["", "\n", "\n\n"]

random.seed(1337)


def _depth(line):
    # count leading 4-space units (bodies are uniformly 4-space indented)
    n = 0
    while line.startswith("    "):
        n += 1
        line = line[4:]
    return n


def reindent(code, indent):
    # code bodies are written with 4-space indentation; rebase onto `indent`
    out = []
    for line in code.split("\n"):
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        depth = _depth(line)
        out.append(indent * depth + stripped)
    return "\n".join(out)


def render(item):
    prose = random.choice(item["prose"])
    sig = random.choice(SIGS)
    header = random.choice(HEADERS)
    indent = random.choice(INDENTS)
    brace = random.choice(OPEN_BRACE_STYLES)
    pre = random.choice(SPACING)
    post = random.choice(SPACING)
    body = reindent(item["code"], indent)
    code = f"{header}{sig}{brace}\n{body}\n}}{post}"
    return f"{prose}\n\n{pre}{code}"


def build():
    # one near-verbatim pass over every item so nothing is missing
    samples = [render(item) for item in ITEMS]
    # many randomized style variants so the model sees wide diversity
    for item in ITEMS:
        for _ in range(6):
            samples.append(render(item))
    return samples


def main():
    samples = build()
    payload = {
        "name": "hello-world-simple",
        "description": (
            "Varied simple hello-world, argv, and basic-arithmetic C programs, "
            "each preceded by a matching one-line prose explanation"
        ),
        "samples": samples,
    }
    out_path = "corpus/hello_corpus.json"
    try:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        raise SystemExit(f"could not write {out_path!r}: {exc}") from exc
    print(f"wrote {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    main()
