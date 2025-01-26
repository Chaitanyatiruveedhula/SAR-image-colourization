from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
import os
from colorizers import *

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

def process_image(img_path, use_gpu=False):
    img = load_img(img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    if use_gpu:
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()
        tens_l_rs = tens_l_rs.cuda()
    
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    return img_bw, out_img_eccv16, out_img_siggraph17

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            img_bw, out_img_eccv16, out_img_siggraph17 = process_image(file_path)

            bw_path = os.path.join(RESULT_FOLDER, 'bw_' + file.filename)
            eccv16_path = os.path.join(RESULT_FOLDER, 'eccv16_' + file.filename)
            siggraph17_path = os.path.join(RESULT_FOLDER, 'siggraph17_' + file.filename)

            plt.imsave(bw_path, img_bw)
            plt.imsave(eccv16_path, out_img_eccv16)
            plt.imsave(siggraph17_path, out_img_siggraph17)

            return render_template('test.html', 
                                   original=file_path, 
                                   bw=bw_path, 
                                   eccv16=eccv16_path, 
                                   siggraph17=siggraph17_path)
    return render_template('test.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
