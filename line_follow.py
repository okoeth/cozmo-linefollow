import asyncio
import cozmo
import cv2
import getopt
import logging
import numpy as np
import sys
import threading
import time

from cozmo.util import degrees, distance_mm, speed_mmps
from matplotlib import pyplot as plt

""" A simple line follow logic, which guides Cozmo along a line.
	For best results, use a white paper and a line that is 5-10mm
	thick. Make sure that no other marks are on the paper, as it is
	currently assumed that only one contour is found in the zone
	used for line position analysis.

	Potential improvements:
	(1) Crive continously and only stop for correction of
		direction
	(2) In case of multiple contours, select best contour 
	    (heuristics could be: in the middle, nearest the last
		contour, ...)
	(3) Master sharp turns
	(4) Master crossings

	Further work: Would be interesting to replace this complex
	logic by just a neural network as done here:
	https://github.com/benjafire/CozmoSelfDriveToyUsingCNN
"""

# Bottom zone where path contour is search in
btm_zn_x = 30
btm_zn_y = 180
btm_zn_w = 260
btm_zn_h = 20

# Constants for corrections
gap_s = 20
gap_m = 60
gap_l = 120
turn_s = 5
turn_m = 15
turn_l = 45
move_s = 5
move_m = 10
move_l = 15
cam_center = 160
step_l = 20

# Show/hide images
show_img=True

# Logger
log = logging.getLogger('ok.linefollow')

def get_path_rect(img):
	""" Finds the bounding rect of the contour of the path. For this
		image is denoised and converted in black and white and inverted.
	"""
	# Prepare image
	raw_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	crp_img = raw_img[btm_zn_y:btm_zn_y+btm_zn_h, btm_zn_x:btm_zn_x+btm_zn_w]
	#cv2.imshow('crp_img', crp_img)
	blr_img = cv2.GaussianBlur(crp_img, (5,5), 0)
	ret,thr_img = cv2.threshold(blr_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#cv2.imshow('thr_img', thr_img)
	inv_img = cv2.bitwise_not(thr_img)
	#cv2.imshow('inv_img', inv_img)
	# Find contour of path
	cnt_img, cnts, h = cv2.findContours(inv_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if len(cnts)==0:
		log.warning('No contour found')
		return 0, 0, 0, 0
	if len(cnts)>1:
		log.warning('More than one contour found: '+str(len(cnts)))
		if show_img:
			wrn_img = cv2.cvtColor(inv_img, cv2.COLOR_GRAY2RGB)
			wrn_img = cv2.drawContours(wrn_img, cnts, -1, (0,255,0), 3)
			cv2.imshow('contour_warning', wrn_img)
	# Find bound rect of path
	x,y,w,h = cv2.boundingRect(cnts[0])
	return x+btm_zn_x,y+btm_zn_y,w,h


def get_path_center(x, w):
	""" Returns the centre of the path
	"""
	return int(round(x+w/2))


def draw_path_rect(img, x, y, w, h, c):
	""" Draw the path rectangle including a vertical line indicating
		the center of the path
	"""
	ret_img = cv2.rectangle(img,(x,y),(x+w,y+h), c, 1)
	ret_img = cv2.line(ret_img, (get_path_center(x, w), y), (get_path_center(x, w),y+h), c, 1)
	return ret_img


def draw_grid(img):
	""" Draws a grid on the image showing zones and center
	"""
	ret_img = cv2.rectangle(img,(btm_zn_x-2,btm_zn_y-2),(btm_zn_x-2+btm_zn_w+4,btm_zn_y-2+btm_zn_h+4), (255,255,0), 1)
	height, width = img.shape[:2]
	center = int(round(width/2))
	ret_img = cv2.line(img, (center, 0), (center,height), (255,255,0), 1)
	return ret_img
	

def move_right(robot: cozmo.robot.Robot, move):
	""" Moves the robot to the right, in case the path is too far on the right
	"""
	log.info('Move right...')
	robot.turn_in_place(degrees(-45)).wait_for_completed()
	robot.drive_straight(distance_mm(move), speed_mmps(5)).wait_for_completed()
	robot.turn_in_place(degrees(45)).wait_for_completed()


def move_left(robot: cozmo.robot.Robot, move):
	""" Moves the robot to the left, in case the path is too far on the right
	"""
	log.info('Move left...')
	robot.turn_in_place(degrees(45)).wait_for_completed()
	robot.drive_straight(distance_mm(move), speed_mmps(5)).wait_for_completed()
	robot.turn_in_place(degrees(-45)).wait_for_completed()


def correct_position(robot: cozmo.robot.Robot, cur_center):
	""" Corrects the position of the robot. Currently, the following corrections
		are implemented (see top of file for constant definitions):
		1) Current path center is >gap_s / >gap_m / >gap_l pixels to the right, 
		   this requires a turn_s / turn_m / turn_l to the left (positive degree)
		2) Current path center is >gap_s / >gap_m / >gap_l pixels to the left, 
		   this requires a turn_s / turn_m / turn_l to the right (positive degree)
	"""
	log.info('Correcting position...')
	gap_abs = cam_center - cur_center
	log.info('Current center  : '+str(cur_center))
	log.info('Absolute gap    : '+str(gap_abs))
	if gap_abs < -gap_l:
		log.info('Current center is gap_l to right, needs right turn_l')
		robot.turn_in_place(degrees(-turn_l)).wait_for_completed()
		return 10
	elif gap_abs < -gap_m:
		log.info('Current center is gap_m to right, needs right turn_m')
		robot.turn_in_place(degrees(-turn_m)).wait_for_completed()
		return 20
	elif gap_abs < -gap_s:
		log.info('Current center is gap_s to right, needs right turn_s')
		robot.turn_in_place(degrees(-turn_s)).wait_for_completed()
		return 30
	elif gap_abs > gap_l:
		log.info('Current center is gap_l to left, needs left turn_l')
		robot.turn_in_place(degrees(turn_s)).wait_for_completed()
		return 10
	elif gap_abs > gap_m:
		log.info('Current center is gap_m to left, needs left turn_m')
		robot.turn_in_place(degrees(turn_m)).wait_for_completed()
		return 20
	elif gap_abs > gap_s:
		log.info('Current center is gap_s to left, needs left turn_s')
		robot.turn_in_place(degrees(turn_s)).wait_for_completed()
		return 30
	else:
		return 40


def step_forward(robot: cozmo.robot.Robot):
	""" Implements the main forward step. The robot takes a step forward and then
		compares the bounding rects of the contour of the path and adjusts direction
		accordingly. First implementation works with small / medium / large adjustments
		only.
	"""
	log.info('Stepping forward...')
	# Take snapshot and store path rect
	log.info('BEGIN_CALC')
	robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE).wait_for_completed()
	raw_img = np.array(robot.world.latest_image.raw_image)
	cur_x, cur_y, cur_w, cur_h = get_path_rect(raw_img)
	# Visualise path rects in second image
	pth_img = draw_grid(raw_img)
	pth_img = draw_path_rect(pth_img, cur_x, cur_y+2, cur_w, cur_h, (0,255,0))
	if show_img:
		cv2.imshow('step_forward', pth_img)
	# Trigger correction
	log.info('END_CALC')
	proposed_step = correct_position(robot, get_path_center(cur_x, cur_w))
	log.info('Moving forward '+str(proposed_step)+'mm')
	robot.drive_straight(distance_mm(proposed_step), speed_mmps(20), False).wait_for_completed()


def capture(robot: cozmo.robot.Robot):
	""" Capture an image of the path and show bounding rect of the 
		contour of the path.
	"""
	log.info('Capture image...')
	robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE).wait_for_completed()
	raw_img = np.array(robot.world.latest_image.raw_image)
	x, y, w, h = get_path_rect(raw_img)
	pth_img = cv2.rectangle(raw_img, (x,y), (x+w,y+h), (255,0,0) ,2)
	cv2.imshow('capture', pth_img)


def battery_level(robot: cozmo.robot.Robot):
	""" Logs the battery level
	"""
	log.info('Battery level...')
	level = robot.battery_voltage
	log.info('Level is '+str(level)+'V')
	if level<=3.5:
		log.warning('Level is low. Please place Cozmo on charger.')


def cozmo_cli(robot: cozmo.robot.Robot):
	""" Main loop implementing simplistic CLI
	"""
	log.info('Entering Cozmo program')
	robot.set_head_light(True)
	robot.camera.image_stream_enabled = True
	robot.set_lift_height(1.0).wait_for_completed()
	while True:
		run_cmd=input('C>')
		if run_cmd == 's':
			step_forward(robot)
		if run_cmd == 'r':
			move_right(robot, move_l)
		if run_cmd == 'l':
			move_left(robot, move_l)
		if run_cmd == 'c':
			capture(robot)
		if run_cmd == 'b':
			battery_level(robot)
		if run_cmd == 'n':
			time.sleep(1)
		if run_cmd == 'e':
			cv2.destroyAllWindows()
			print('Bye.')
			break


def cozmo_step_forward(robot: cozmo.robot.Robot):
	""" Follows line forever
	"""
	log.info('Entering Cozmo step_forward...')
	robot.set_head_light(True)
	robot.camera.image_stream_enabled = True
	robot.set_lift_height(1.0).wait_for_completed()
	while True:
		step_forward(robot)


def cozmo_battery_level(robot: cozmo.robot.Robot):
	""" Output battery level
	"""
	log.info('Entering Cozmo battery_level...')
	battery_level(robot)


def main(argv):
	# Set-up logging
	formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s %(message)s')
	handler = logging.StreamHandler()
	handler.setLevel(logging.INFO)
	handler.setFormatter(formatter)
	log.setLevel(logging.INFO)
	log.addHandler(handler)
	# Eval command line
	usecase='cli'
	try:
		opts, args = getopt.getopt(argv,'hu:',['usecase='])
	except getopt.GetoptError:
		print('line_follow.py -u <usecase>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('line_follow.py -u <usecase>')
			sys.exit()
		elif opt in ("-u", "--usecase"):
			usecase=arg
	log.info('Executing use case '+usecase)
	if usecase=='cli':
		cozmo.run_program(cozmo_cli)
	elif usecase=='step_forward':
		cozmo.run_program(cozmo_step_forward)
	elif usecase=='battery_level':
		cozmo.run_program(cozmo_battery_level)

if __name__ == '__main__':
	main(sys.argv[1:])