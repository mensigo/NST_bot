import os
import logging

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.types.message import ContentType
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup

from config import (API_TOKEN, PROXY_URL, PROXY_AUTH,
				   WEBHOOK_URL, WEBAPP_HOST, WEBAPP_PORT)
from model.image_loader import *
from model.losses import *
from model.model import *
from model.gan_test import *

WEBHOOK_USAGE_FLG = False
PROXY_USAGE_FLG = True

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN) if not PROXY_USAGE_FLG \
	  else Bot(token=API_TOKEN, proxy=PROXY_URL, proxy_auth=PROXY_AUTH)
dp = Dispatcher(bot, storage=MemoryStorage())

class RunStates(StatesGroup):
    content = State()
    style = State()
    intensity = State()
    waiting = State()
    run_same = State()

class UkiyoeStates(StatesGroup):
	content = State()
	waiting = State()

######################################
##      Basic message handlers      ##
######################################

@dp.message_handler(commands=['start'], state='*')
async def handler_command_start(message: types.Message):
    
	msg = \
		'''
		Greetings1
		I\'m summoned to this world to help you transfer styles from one
		picture to another1 (i know you got them c;)
		So here are the magic words to communicate w/ me:

		/run - get instructions to do style transfer (your style)

		/ukiyoe - same but fixed (ukiyoe) style

		/stopit - return to the starting point

		/help - about me & procedure
		'''.replace('\t','')
	logging.info('Started.')
	await message.answer(msg)


@dp.message_handler(commands=['help'], state="*")
async def handler_command_help(message: types.Message):

	msg = \
		'''	
		I'm a bot designed to transfer artistic styles from style photos
		to content photos. To do so i need you to:
		  a) enter /run
		  b) send photo to transfer style on [content photo]
		  c) send photo to get style from [style photo]
		  d) wait while I'll be working hard (approx. a minute)
		  e) enjoy result image (I'll send)

		Also there is an option for transferring ukiyoe style (/ukiyoe).

		- if you want to perform something beautiful, enter /run or /ukiyoe
		  and follow my orders
		- if you change your mind and want to start over (at any time),
		  enter /stopit

		ps. I can\'t handle pics with the largest side bigger than 256 pix
			(it will be resized automatically)

		[made by @mensigo]
		'''.replace('\t','')
	await message.answer(msg)


@dp.message_handler(commands=['stopit'], state='*')
async def handler_command_stopit(message: types.Message, state: FSMContext):

	logging.info('State is reset.')
	await state.finish()
	await message.answer('All right, let\'s try again from the very beginning. \nEnter /help',
						 reply_markup=types.ReplyKeyboardRemove())

######################################


@dp.message_handler(commands=['run'], state=None)
async def handler_command_run(message: types.Message):

	msg = 'First, give me content image to transfer style on.'
	logging.info('Content state is set.')
	await RunStates.content.set()
	await message.answer(msg)


@dp.message_handler(lambda message: message.content_type != ContentType.PHOTO,
					state=RunStates.content)
async def handler_content_invalid(message: types.Message, state: FSMContext):

	msg = 'I can\'t read it, send content image pls.'
	return await message.reply(msg)

@dp.message_handler(lambda message: message.content_type != ContentType.PHOTO,
					state=RunStates.style)
async def handler_style_invalid(message: types.Message, state: FSMContext):

	msg = 'I can\'t read it, send style image pls.'
	return await message.reply(msg)


@dp.message_handler(state=RunStates.content, content_types=ContentType.PHOTO)
async def handler_content(message: types.Message, state: FSMContext):

	logging.info('Style state is set.')
	await RunStates.style.set()
	await state.update_data(content_id=message.photo[-1].file_id)
	await message.answer('Nice1 Now send me an image to get style from.')


@dp.message_handler(state=RunStates.style, content_types=ContentType.PHOTO)
async def handler_style(message: types.Message, state: FSMContext):

	logging.info('Intensity state is set.')
	await RunStates.intensity.set()
	await state.update_data(style_id=message.photo[-1].file_id)

	markup = types.ReplyKeyboardMarkup(resize_keyboard=True,
									   selective=True,
									   one_time_keyboard=True)
	markup.add('Light', 'Medium', 'Hard')
	await message.answer('At last, choose style intensity.', reply_markup=markup)


######################################
##          Style transfer          ##
######################################

async def get_output(input_data, message: types.Message):

	user_id = message.from_user.id
	content_path = str(user_id) + '_content.jpg'
	style_path = str(user_id) + '_style.jpg'
	output_path = str(user_id) + '_output.jpg'

	await bot.download_file_by_id(input_data['content_id'], destination=content_path)
	await bot.download_file_by_id(input_data['style_id'], destination=style_path)
	await apply_NST(content_path, style_path, output_path, input_data['style_w'])

	with open(output_path, 'rb') as output_img:
		await message.answer_photo(output_img, caption='Ta-daa1\n')


@dp.message_handler(lambda message: message.text not in ['Light', 'Medium', 'Hard'],
					state=RunStates.intensity)
async def handler_intensity_invalid(message: types.Message):

	return await message.reply("Wrong option.\nChoose intensity from the keyboard below.")


@dp.message_handler(state=RunStates.intensity)
async def handler_intensity(message: types.Message, state: FSMContext):

	if (message.text == 'Light'):
		style_w = 1e5
	elif (message.text == 'Medium'):
		style_w = 1e7
	elif (message.text == 'Hard'):
		style_w = 1e9
	else:
		loggind.error('Wrong intensity string.')
		await message.answer('Wrong option.\nMake sure, you used the special buttons.')
		return

	logging.info('Intensity option is set to {} ({}).'.format(style_w, message.text))
	markup = types.ReplyKeyboardRemove()
	await state.update_data(style_w=style_w)
	
	logging.info('Waiting state is set.')
	await RunStates.waiting.set()
	await message.answer('All right. Now wait a minute..')
	await types.ChatActions.typing()

	input_data = await state.get_data()
	await get_output(input_data, message)

	# run with same photos
	
	logging.info('Run_same state is set.')
	await RunStates.run_same.set()
	markup = types.ReplyKeyboardMarkup(resize_keyboard=True,
									   selective=True,
									   one_time_keyboard=True)
	markup.add('Yes!', 'Nope.')
	await message.answer('Try another intensity with same images?',
						 reply_markup=markup)


@dp.message_handler(lambda message: message.text not in ['Yes!', 'Nope.'],
					state=RunStates.run_same)
async def handler_run_same_invalid(message: types.Message, state: FSMContext):

	return await message.reply("Wrong option.\nChoose from the keyboard below.")


@dp.message_handler(state=RunStates.run_same)
async def handler_run_same(message: types.Message, state: FSMContext):

	if (message.text == 'Yes!'):

		# change intensity

		logging.info('Intensity state is set.')
		await RunStates.intensity.set()
		markup = types.ReplyKeyboardMarkup(resize_keyboard=True,
										   selective=True,
										   one_time_keyboard=True)
		markup.add('Light', 'Medium', 'Hard')
		await message.answer('Choose new intensity.', reply_markup=markup)

	else:

		# finish

		logging.info('Finished.')
		await state.finish()
		await message.answer('Ok, you can /run whenever you want c;',
							 reply_markup=types.ReplyKeyboardRemove())

		user_id = message.from_user.id
		content_path = str(user_id) + '_content.jpg'
		style_path = str(user_id) + '_style.jpg'
		output_path = str(user_id) + '_output.jpg'

		for p in [content_path, style_path, output_path]:
			os.remove(p)


######################################
##          Ukiyoe transfer         ##
######################################

@dp.message_handler(commands=['ukiyoe'], state='*')
async def handler_command_ukiyoe(message: types.Message):

	msg = \
		'''
		Send me content image to transfer style on.
		(a squared image would be perfect)
		'''.replace('\t','')
	logging.info('Content state is set.')
	await UkiyoeStates.content.set()
	await message.answer(msg)


@dp.message_handler(lambda message: message.content_type != ContentType.PHOTO,
					state=UkiyoeStates.content)
async def handler_content_ukiyoe_invalid(message: types.Message, state: FSMContext):

	msg = 'I can\'t read it, send content image pls.'
	return await message.reply(msg)


async def get_output_ukiyoe(input_data, message: types.Message):

	user_id = message.from_user.id
	content_path = str(user_id) + '_content.jpg'
	output_path = str(user_id) + '_output.png'

	await bot.download_file_by_id(input_data['content_id'], destination=content_path)
	await apply_GAN(content_path, output_path)

	with open(output_path, 'rb') as output_img:
		await message.answer_photo(output_img, caption='Awesome1\n')
	os.remove(output_path)


@dp.message_handler(state=UkiyoeStates.content, content_types=ContentType.PHOTO)
async def handler_content_ukiyoe(message: types.Message, state: FSMContext):

	logging.info('Waiting state is set.')
	await UkiyoeStates.waiting.set()
	await state.update_data(content_id=message.photo[-1].file_id)
	await message.answer('All right. Now wait a bit..')
	await types.ChatActions.typing()

	input_data = await state.get_data()
	await get_output_ukiyoe(input_data, message)

	logging.info('Finished.')
	await state.finish()
	await message.answer('So.. you can /run whenever you want c;')

######################################


@dp.message_handler(state="*", content_types=ContentType.ANY)
async def handler_other_msg(message: types.Message):

	return await message.answer('Incorrect action. Enter /help')


async def startup(dp: Dispatcher):
	logging.warning('Starting..')
	if (WEBHOOK_USAGE_FLG):
		await bot.set_webhook(WEBHOOK_URL)


async def shutdown(dp: Dispatcher):

	# just in case, remove excess images
	for fname in os.listdir():
		if (fname.endswith('.jpg') or fname.endswith('.png')):
			os.remove(fname)
	await dp.storage.close()
	await dp.storage.wait_closed()
	logging.warning('Bye!')


if __name__ == '__main__':

	if (WEBHOOK_USAGE_FLG):
		executor.start_webhook(
			dispatcher=dp,
			webhook_path=WEBHOOK_PATH,
			on_startup=startup,
			on_shutdown=shutdown,
			skip_updates=False,
			host=WEBAPP_HOST,
			port=WEBAPP_PORT,
		)
	else:
		executor.start_polling(
			dispatcher=dp,
			skip_updates=False,
			on_startup=startup,
			on_shutdown=shutdown
		)