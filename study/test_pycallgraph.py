from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput

def foo():
    for i in range(5):
        print(i)

def bar():
    foo()

graphviz = GraphvizOutput()
graphviz.output_file = 'basic.png'

with PyCallGraph(output=graphviz):
    bar()
