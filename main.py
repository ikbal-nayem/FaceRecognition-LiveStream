from flask import Flask, request
from FaceRecognition import applyWithURL, applyWithImg

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def recognizer():
    if request.method == "POST":
        if request.is_json:
            url = request.get_json().get('url', False)
            faces = applyWithURL(url) if url else {'success': False,'message': "Image url not provided into 'url'"}
            return(faces)
        elif request.files:
            img = request.files.get('image', False)
            faces = applyWithImg(img) if img else {'success': False,'message': "Image wasn't provided into 'image'"}
            return(faces)
        else:
            return({'success': False, 'message': 'Request data should be in JSON format'})
    else:
        return("Get request")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)