import numpy as np
import cv2




def robust_LS(A,y,iter=10) :
    A_plus = np.linalg.inv(A.T @ A) @ A.T
    p = A_plus @ y

    params =[]
    points = []
    for _ in range(iter) :
        r = y - A @ p
        W = np.diag(1 / (np.abs(r)/1.3998 + 1))
        p = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
        a, b = p
        x1, x2 = 0, 640*2
        y1, y2 = int(a*x1+b), int(a*x2+b)
        params.append(p)
        points.append(((x1,y1), (x2, y2)))

    return params, points


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
            l_m = np.array(p1)
            r_m = np.array(p2)

            # 일반적인 LS
            A_plus = np.linalg.inv(l_m.T @ l_m) @ l_m.T
            result = A_plus @ r_m
            a,b = result
            cv2.putText(window, f'LS : y={a:.4f}x+{b:.4f}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 1, cv2.LINE_AA)

            x1, x2 = 0, 640*2
            y1, y2 = int(a*x1+b), int(a*x2+b)
            cv2.line(window, (x1,y1), (x2,y2), (0,0,255), 1, cv2.LINE_AA)

            #Robust LS

            params, points = robust_LS(l_m, r_m, 10)
            
            for i, ps in enumerate(points): 
                a, b = params[i]
                color = [(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,255)]

                cv2.putText(window, f'Robust_{i} : y={a:.4f}x+{b:.4f}', (10, 20 + (i+1)*30), cv2.FONT_HERSHEY_PLAIN, 1.0, color[i%10], 1, cv2.LINE_AA)
                cv2.line(window, ps[0], ps[1], color[i%10], 1, cv2.LINE_AA)
            cv2.imshow(title, window)
            draw_on = False


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

title = 'Fit Line'
cv2.namedWindow(title)
cv2.setMouseCallback(title, on_mouse, window)
cv2.imshow(title, window)
cv2.waitKey()
cv2.destroyAllWindows()