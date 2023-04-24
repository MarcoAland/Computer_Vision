import cv2
import mediapipe as mp
import time

# class creation
class poseDetector():
    def __init__(self, mode=False, modelComplexity=1, smooth=True,segmentation=False,smoothSegment=True,detectionCon=0.5,trackingCon=0.5):
        self.mode = mode
        self.modelComplex = modelComplexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.smoothsegment  = smoothSegment
        
        
        self.detectionCon = detectionCon
        self.trackCon = trackingCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplex, self.smooth, self.segmentation, self.smoothsegment, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # it gives small dots onhands total 32 landmark points

    def findPose(self,img,draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB) # process the frame
        #print(results.multi_hand_landmarks)

        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findPosition(self,img, draw=True):
        """Lists the position/type of landmarks
        we give in the list and in the list ww have stored
        type and position of the landmarks.
        List has all the lm position"""

        lmlist = []

        # check wether any landmark was detected
        if self.results.pose_landmarks:
            # Get id number and landmark information
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # id will give id of landmark in exact index number
                # height width and channel
                h,w,c = img.shape
                #find the position
                cx,cy = int(lm.x*w), int(lm.y*h) #center
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])

        return lmlist   

def main():
    #Frame rates
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0) # cv2.VideoCapture(PoseVideos/{video.mp4})
    detector = poseDetector()

    while True:
        success,img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        # print(lmList) to print list of postition each id
        if len(lmList) != 0:
            print(lmList[0]) # Number of index can be changed

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f"FPS {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (170, 255, 0), 4) # (video, text, location on video, font, size, color, thickness)

        cv2.imshow("Video",img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()