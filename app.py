from flask import Flask, render_template, request, redirect, url_for
import os
from webapp.components.deepdream.deepdream import DeepDreamGenerator
from webapp.components.gan.gan import GanImageGenerator

app = Flask(__name__)
UPLOAD_FOLDER = 'static/images/uploaded'
DISPLAY_FOLDER = 'images/generated'
MODEL_FOLDER = 'webapp/models/'
UPLOAD_DISPLAY_FOLDER = 'images/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DISPLAY_FOLDER'] = DISPLAY_FOLDER
app.config['UPLOAD_DISPLAY_FOLDER'] = UPLOAD_DISPLAY_FOLDER


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/uploadImage', methods=['POST'])
def uploadImage():
    # print('Inside Uploading image')
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Retrieve the model selection from the form
            selected_model = request.form.get('model', 'DeepDream')
            print(f'Selected model: {selected_model}')

            if selected_model == 'DeepDream':
                generator = DeepDreamGenerator(filepath)
                generated_filepath, score = generator.generate_dream()

            if selected_model == 'GAN1':
                model_path = MODEL_FOLDER + 'gan1.keras'
                model_absolute_path = os.path.abspath(model_path)

                generator = GanImageGenerator(model_absolute_path, filepath)
                print('Generator: GAN1')
                generated_filepath, score = generator.generate_gan_image()

            if selected_model == 'GAN2':
                model_path = MODEL_FOLDER + 'gan2.keras'
                model_absolute_path = os.path.abspath(model_path)

                generator = GanImageGenerator(model_absolute_path, filepath)
                print('Generator: GAN2')
                generated_filepath, score = generator.generate_gan_image()

            filename = os.path.basename(generated_filepath)
            # print(generated_filename)
            # print(file.filename)
            # return redirect(url_for('display', filename=file.filename))
            return redirect(url_for('display', filename=filename, score=score, orig_img=file.filename))
    return render_template('upload.html')


@app.route('/display')
def display():
    filename = request.args.get('filename', default=None)
    score = request.args.get('score', default=None)
    orig_img = request.args.get('orig_img', default=None)
    image_url = os.path.join(app.config['DISPLAY_FOLDER'], filename)
    uploaded_image_url = os.path.join(
        app.config['UPLOAD_DISPLAY_FOLDER'], orig_img)
    return render_template('display.html', image_url=image_url, score=score, uploaded_image_url=uploaded_image_url)


if __name__ == '__main__':
    app.run(debug=True)
