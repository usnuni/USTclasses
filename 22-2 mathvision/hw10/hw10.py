import numpy as np
import cv2

def on_mouse(click, x, y, flags, param):

    global points, p1, p2, p3, draw_on, window

    
    if click == cv2.EVENT_LBUTTONDOWN and draw_on: 
        points.append((x,y))
        p1.append((x,1))
        p2.append((y))
        p3.append((x,y,1))
        return False

    elif click == cv2.EVENT_LBUTTONUP and draw_on:
        cv2.line(window, points[-1], points[-1], (255,255,255), 5)  
        cv2.imshow(title, window)
        
        print(f'points : {points}') 

    elif click == cv2.EVENT_RBUTTONDOWN :
        
        if len(points) >= 2 :
            p1_l = np.array(p1)
            p2_r = np.array(p2)

            # Pseudo inverse
            pi = np.linalg.inv(p1_l.T @ p1_l) @ p1_l.T
            result = pi @ p2_r
            a,b = result
            cv2.putText(window, f'LS_1 : y={a:.4f}x+{b:.4f}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0))
            x1, x2 = 0, 640
            y1, y2 = int(a*x1+b), int(a*x2+b)
            cv2.line(window, (x1,y1), (x2,y2), (0,255,0), 1)

            # SVD
            U,D,V = np.linalg.svd(np.array(p3))  
            a,b,c = V[-1]
            cv2.putText(window, f'LS_2 : {a:.4f}x+{b:.4f}y+{c:.4f}=0', (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255))
            if np.abs(b) < 1e-9 :
                x1 = -c/a
                cv2.line(window, (x1,0), (x2,H), (0,0,255), 1)
            else :
                y1 = int((-c - (a*x1))/ b)
                y2 = int((-c - (a*x2))/ b)
                cv2.line(window, (x1,y1), (x2,y2), (0,0,255), 1)
            cv2.imshow(title, window)


    elif click == cv2.EVENT_LBUTTONDBLCLK : # reset drawing
        points = []
        p1 = []
        p2 = []
        p3 = []
        window = np.ones((480, 640, 3)) *0
        cv2.imshow(title, window)
    

draw_on = True
points = []
p1 = []
p2 = []
p3 = []


window = np.ones((480, 640, 3), dtype=np.uint8) * 0

title = 'SVD'
cv2.namedWindow(title)
cv2.setMouseCallback(title, on_mouse, window)
cv2.imshow(title, window)
cv2.waitKey()
cv2.destroyAllWindows()
