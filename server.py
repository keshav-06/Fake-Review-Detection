from flask import Flask, render_template, request, jsonify
import deploy

app = Flask (__name__)

@app.route ('/')
def homepage ():
    return render_template ('template.html')


@app.route ('/api/bot', methods = ['POST'])
def say_name ():
	json = request.get_json()

	review_text = request.form['review_text']
	rating = request.form['rating']
	verified_purchase =  request.form['verified_purchase']
	product_category = request.form['product_category']

	result =  deploy.get_result(review_text, rating, verified_purchase, product_category)
	print (result)

	if result[0] == 1:
		return jsonify (result = 'T')

	else:
		return jsonify (result = 'F')

from pyngrok import ngrok

def config_ngrok() :
    app.config['START_NGROK'] = True
    url = ngrok.connect(5000)
    print(' * Public URL:', url)

if __name__ == '__main__':
    config_ngrok()
    app.run ()
