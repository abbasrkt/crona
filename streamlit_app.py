import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from modelvgg import model

# Add in location to select image.

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
##########
##### Set up sidebar.
##########
# picture = st.camera_input("Take a picture")
#
# if picture:
#      st.image(picture)

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=True)

# st.sidebar.write('[Find additional images on Roboflow.](https://public.roboflow.com/object-detection/bccd/)')

## Add in sliders.
# confidence_threshold = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.01)
# overlap_threshold = st.sidebar.slider('Overlap threshold', 0.0, 1.0, 0.5, 0.01)

# image = Image.open('./images/roboflow_logo.png')
# st.sidebar.image(image,
#                  use_column_width=True)

image = Image.open('./images/corona.jpg')
st.sidebar.image(image, use_column_width=True)

##########
##### Set up main app.
##########

## Title.
st.write('# CORONA Virus Recognition')
latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.01)
## Pull in default image or user-selected image.
sump = [0, 0, 0]
allp = []
if uploaded_file is None or uploaded_file == []:
    # Default image.
    # url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    # image = Image.open("D:\streamlit-bccd-master\streamlit-bccd-master\BCCD_sample_images/test.jpg")
    st.write("please upload your lung image")

else:
    # User-selected image.
    for i in uploaded_file:
        image = Image.open(i)
        newsize = (200, 200)
        image = image.resize(newsize)
        ## Subtitle.
        st.write('### Inferenced Image')
        st.image(image, use_column_width=False)
        # Convert to JPEG Buffer.
        # buffered = io.BytesIO()
        # image.save(buffered, quality=90, format='JPEG')
        imgArray = np.array(image)
        imgArray = cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGB)
        imgArray = imgArray / 255
        imgArray = np.expand_dims(imgArray, axis=0)
        p = model.predict(imgArray)
        plistlib = p.tolist()
        allp.append(plistlib[0])
        zipped_lists = zip(sump, plistlib[0])
        sump = [x + y for (x, y) in zipped_lists]

        stats = {"ImageName": i.name, "NonCovid": "{:.8f}".format(float(plistlib[0][0])),
                 "Covid": "{:.8f}".format(float(plistlib[0][1])),
                 "Normal": "{:.8f}".format(float(plistlib[0][2]))}
        output_dict = {'predictions': stats}
        st.write(output_dict)
        st.write('### JSON Output')

        st.write(json.dumps(plistlib))
    allpNPArray = np.array(allp)
    st.write("mean of all")
    # avgp = [x / uploaded_file.__len__() for x in sump]
    st.write(np.mean(allpNPArray, axis=0))

    st.write("amax of all")
    st.write(np.amax(allpNPArray, axis=0))

    st.write("amin of all")
    st.write(np.amin(allpNPArray, axis=0))



    # Base 64 encode.
    # img_str = base64.b64encode(buffered.getvalue())
    # img_str = img_str.decode('ascii')
    #
    # ## Construct the URL to retrieve image.
    # upload_url = ''.join([
    #     'https://infer.roboflow.com/rf-bccd-bkpj9--1',
    #     '?access_token=vbIBKNgIXqAQ',
    #     '&format=image',
    #     f'&overlap={overlap_threshold * 100}',
    #     f'&confidence={confidence_threshold * 100}',
    #     '&stroke=2',
    #     '&labels=True'
    # ])
    #
    # ## POST to the API.
    # r = requests.post(upload_url,
    #                   data=img_str,
    #                   headers={
    #     'Content-Type': 'application/x-www-form-urlencoded'
    # })
    #
    # image = Image.open(BytesIO(r.content))
    #
    # # Convert to JPEG Buffer.
    # buffered = io.BytesIO()
    # image.save(buffered, quality=90, format='JPEG')

    # Display image.

    ## Construct the URL to retrieve JSON.
    # upload_url = ''.join([
    #     'https://infer.roboflow.com/rf-bccd-bkpj9--1',
    #     '?access_token=vbIBKNgIXqAQ'
    # ])
    #
    # ## POST to the API.
    # r = requests.post(upload_url,
    #                   data=img_str,
    #                   headers={
    #     'Content-Type': 'application/x-www-form-urlencoded'
    # })

    ## Save the JSON.

    ## Generate list of confidences.
    # confidences = [box['confidence'] for box in output_dict['predictions']]

    ## Summary statistics section in main app.
    # st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
    # st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')
    ## Histogram in main app.
    # st.write('### Histogram of Confidence Levels')
    # fig, ax = plt.subplots()
    # ax.hist(confidences, bins=10, range=(0.0,1.0))
    # st.pyplot(fig)

    ## Display the JSON in main app.

    # st.write(r.json())
