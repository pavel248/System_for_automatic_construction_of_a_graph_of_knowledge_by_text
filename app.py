from flask import Flask, render_template, request, redirect, url_for
from modules.main_module import get_graph

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def form_example():
    if request.method == 'POST':
        text = request.form['user_text']
        print(text)
        get_graph(text)
        return redirect('graph')

    return render_template("Index.html")


@app.route('/graph')
def aboba():
    return render_template("Graph_for_group_0.html")


if __name__ == '__main__':
    app.run()
