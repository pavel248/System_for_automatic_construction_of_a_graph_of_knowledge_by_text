import os

from flask import Flask, render_template, request
from modules.mainModule import get_graph

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        text = request.form['user_text']
        print(text)
        get_graph(text)

        return render_template("Graph_for_group_0.html")

    # otherwise handle the GET request
    return render_template("Index.html")


if __name__ == '__main__':
    app.run()
