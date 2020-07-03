import aiohttp

API_TOKEN = '...'
PROXY_URL = '...'
PROXY_AUTH = aiohttp.BasicAuth(login='...', password='...')

# webhook settings
WEBHOOK_HOST = '...'
WEBHOOK_PATH = f'/home/ubuntu/webhook{API_TOKEN}'

# webserver settings
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = 3001