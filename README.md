# step 1
# create a virtual environment here

run: py -m venv venv

# step 2
# activate the virtual environment
venv\Scripts\activate

# step 3
# install the required packages
pip install -r requirements.txt

# step 4
# set envirionment variable
create a .env file in the same directory where requirements.txt is present and add the variables:

# step 5
# run the application fastapi
uvicorn backend.app.main:app --reload


need to use logger now inplace of print
use in below structure:


logger.info("✅ App started.")

logger.info(f"AWS Region: {env_config.get('AWS_REGION')}")
# where env_config.get('AWS_REGION') is variable and need to write inside {}

# for error log
logger.error(f""Error: {e}")



### about logger function

from logging.handlers import TimedRotatingFileHandler

TimedRotatingFileHandler(
    filename='app.log',     # Log file name
    when='midnight',        # When to rotate
    interval=1,             # Rotate every 1 unit of 'when'
    backupCount=7,          # Keep 7 rotated files (older files are deleted)
    encoding='utf-8'        # Encoding of the log file
)


🔹 Parameter: when
This determines how often the logs rotate. Here are the valid options:


| Value        | Meaning                         |
| ------------ | ------------------------------- |
| `'S'`        | Seconds                         |
| `'M'`        | Minutes                         |
| `'H'`        | Hours                           |
| `'D'`        | Days                            |
| `'W0'-'W6'`  | Weekday (0 = Monday)            |
| `'midnight'` | Rotate at midnight (local time) |


🔹 Parameter: interval
It tells how many units of the when value to wait before rotating.

Example:

python
Copy
Edit
when='H', interval=3
→ Rotate the log every 3 hours

python
Copy
Edit
when='midnight', interval=1
→ Rotate every midnight
