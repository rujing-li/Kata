from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/videos/<id>')
def video(id):
    return render_template('video.html', id = id)

@app.route('/trainings/<id>')
def training(id):
    return render_template('training.html', id = id)

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)