from flask import Flask, render_template, request
from calc import calculate_real_size
from db import save_to_db

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        username = request.form['username']
        image_size = float(request.form['image_size'])
        magnification = float(request.form['magnification'])

        result = calculate_real_size(image_size, magnification)
        save_to_db(username, image_size, magnification, result)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
