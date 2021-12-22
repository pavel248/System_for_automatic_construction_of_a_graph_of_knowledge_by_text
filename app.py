from flask import Flask, render_template, request
from modules.mainModule import get_graph



app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return "Hi"

@app.route('/form-example', methods=['GET', 'POST'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        text = request.form['text']
        print(text)
        get_graph(text)

        return render_template("Graph_for_group_0.html")

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>text: <input type="text" name="text"></label></div>
               <input type="submit" value="Submit">
           </form>'''


if __name__ == '__main__':
    app.run()
