import os
import logging

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.types.message import ContentType
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup

from config import API_TOKEN, PROXY_URL, PROXY_AUTH
from model.image_loader import *
from model.losses import *
from model.model import *


logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN, proxy=PROXY_URL, proxy_auth=PROXY_AUTH)
dp = Dispatcher(bot, storage=MemoryStorage())

class RunStates(StatesGroup):
    content = State()
    style = State()
    intensity = State()
    waiting = State()


######################################
##      Basic message handlers      ##
######################################

@dp.message_handler(commands=['start'], state=None)
async def handler_command_start(message: types.Message):
    
	msg = \
		'''
		Greetings1
		I\'m summoned to this world to help you transfer styles from one
		picture to another1 (i know you got them c;)
		So here are the magic words to communicate w/ me:

		/run - get instructions to do style transfer

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

		- if you want to perform something beautiful, enter /run and
		  follow my orders
		- if you change your mind and want to start over (at any time),
		  enter /stopit

		[made by @mensigo]
		'''.replace('\t','')
	await message.answer(msg)


@dp.message_handler(commands=['stopit'], state='*')
async def handler_command_stopit(message: types.Message, state: FSMContext):

	logging.info('State is reset.')
	await state.finish()
	await message.answer('All right, let\'s try again from the very\nbeginning. Enter /run')


@dp.message_handler(commands=['run'], state=None)
async def handler_command_run(message: types.Message):

	msg = \
		'''	
		First, give me content image to transfer style on.
		ps. I can\'t handle pics with the largest side bigger
		       than 300 pix yet (it will be resized automatically)
		'''.replace('\t','')
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
		await message.answer_photo(output_img, caption='Ta-daa1\nYou can /run again, if you want c;')

	for p in [content_path, style_path, output_path]:
		os.remove(p)


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
	logging.info('Finished.')
	await state.finish()


@dp.message_handler(state="*", content_types=ContentType.ANY)
async def handler_other_msg(message: types.Message):

	return await message.answer('Incorrect action. Enter /help')


async def shutdown(dp: Dispatcher):
	await dp.storage.close()
	await dp.storage.wait_closed()


if __name__ == '__main__':

	executor.start_polling(dp, skip_updates=True, on_shutdown=shutdown)