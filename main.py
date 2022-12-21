import numpy as np
import cv2 as cv

models_path = '/opt/homebrew/Cellar/opencv/4.6.0_1/share/opencv4/haarcascades/'

face_cascade = cv.CascadeClassifier()


def zoom_at(img, zoom, coord=None):
    # Translate to zoomed coordinates
    h, w, _ = [ zoom * i for i in img.shape ]
    
    if coord is None: x, y = w/2, h/2
    else: x, y = [ zoom*c for c in coord ]

    box_w, box_h = w/zoom, h/zoom

    start_x_unclipped = int(round(x - box_w/2))
    start_y_unclipped = int(round(y - box_h/2))

    start_x_unclipped = max(0, start_x_unclipped); start_x = min(int(w), start_x_unclipped + int(box_w)) - int(box_w)
    start_y_unclipped = max(0, start_y_unclipped); start_y = min(int(h), start_y_unclipped + int(box_h)) - int(box_h)

    img = cv.resize( img, (0, 0), fx=zoom, fy=zoom)

    img_ret = img[ 
        start_y : start_y + int(box_h),
        start_x : start_x + int(box_w),
    :]

    # print("bounding box: ", start_x, start_y, end_x, end_y)
    # print("zoomed image shape: ", img_ret.shape)

    box = np.array([start_x, start_y, box_w, box_h])
    box = (box / zoom).astype(int)

    return img_ret, box

def get_faces(frame: cv.Mat):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = np.array(face_cascade.detectMultiScale(frame_gray), dtype=int)
    return faces

def get_box_roi(frame: cv.Mat, box: np.ndarray):
    x, y, w, h = box

    center = (x + w//2, y + h//2)

    zoom = max(min(frame.shape[1]/w, frame.shape[0]/h, 10), 1)
    return zoom_at(frame, zoom, center)

def get_box(box: np.ndarray):
    return (box[0], box[1], box[0]+box[2], box[1]+box[3])

def render(frame: cv.Mat, mode: int, **kwargs):
    if mode == 1 and 'roi' in kwargs:
        # Render zoomed image on face
        cv.imshow('Render', kwargs['roi'])
    elif mode == 2 and 'roi' in kwargs:
        # Render original frame with bounding box
        roi_box = kwargs['roi_box']
        if 'face' in kwargs:
            face_box = kwargs['face']
            cv.rectangle(frame, face_box ,(255, 0, 0), 2)
        cv.rectangle(frame, roi_box , (0, 255, 0), 2)
        cv.imshow('Render', frame)
    elif mode == 3 and 'prob' in kwargs:
        # Render back projection probability map
        frame[:] = kwargs['prob'][...,np.newaxis]
        cam_shift_box = kwargs['cam_shift_box']
        cam_shift_ellipse = kwargs['cam_shift_ellipse']
        cv.rectangle(frame, cam_shift_box ,(255, 255, 0), 2)
        cv.ellipse(frame, cam_shift_ellipse, (255, 0, 255), 2)
        cv.imshow('Render', frame)
    else:
        cv.imshow('Render', frame)

def live_video():
    cap = cv.VideoCapture(0)
    target_box = prev_box = np.array([1,1,1,1], dtype=int)
    mode = 2
    i = 0
    hist = None
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 100, .1 )

    while(True):
        ret, frame = cap.read()
        render_data = {}
        if ret:
            frame = cv.flip(frame, 1)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 150., 255.)))

            # once every 100 frames
            if i % 100 * 2 < 10 or hist is None:
                faces = get_faces(frame)
                if faces.shape[0] > 0:
                    x, y, w, h = faces[0]

                    target_box = np.array([x, y, w + w//2, h + h//2], dtype=int)
                    for face in faces:
                        x, y, w, h = face

                        hsv_roi = hsv[y:y+h, x:x+w]
                        mask_roi = mask[y:y+h, x:x+w]

                        new_hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                        cv.normalize(new_hist, new_hist, 0, 255, cv.NORM_MINMAX)
                        new_hist = new_hist.reshape(-1)

                        # the hist np vector is the weighted sum of hist and new_hist by 0.1 
                        hist = new_hist if hist is None else hist*0.9 + new_hist*0.1
                    
                    render_data['face'] = faces[0]


            if hist is not None:
                prob = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)
                prob &= mask

                track_box, target_box = cv.CamShift(prob, target_box, term_crit)
                target_box = np.array(target_box)

                alpha = .1
                current_box = (prev_box * (1 - alpha) + target_box * alpha).astype(int)
                np.copyto(prev_box, current_box)

                render_data['prob'] = prob
                render_data['cam_shift_box'],  render_data['cam_shift_ellipse'] = target_box, track_box
                render_data['roi'], render_data['roi_box'] = get_box_roi(frame, current_box)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                mode = 1
            elif key == ord('2'):
                mode = 2
            elif key == ord('3'):
                mode = 3
    
            render(frame, mode, **render_data)
            i += 1

    cap.release()
    cv.destroyAllWindows()

def main():
    face_cascade.load(models_path + 'haarcascade_frontalface_alt.xml')
    live_video()

if __name__ == '__main__':
    main()

