from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
import telegram
import logging
import os
from credentials import token
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from connections import get_db_session
from model import Queries
import time

# init
bot = telegram.Bot(token)
session = get_db_session(expire_on_commit=False)
updater = Updater(token, use_context=True)
logging.basicConfig(filename='bot.logs', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

current_path = os.path.dirname(os.path.realpath(__file__))


def sharp(update, context):
    query = update.callback_query
    query.answer()
    chat, msg_text, N_YEARS = query.data.split('_')

    query.edit_message_text(text='You have choose {} year(s) interval. Wait for data.'.format(N_YEARS))

    N_YEARS = int(N_YEARS)

    db_write_queries(
        [
            chat,
            '',
            '',
            '',
            msg_text])

    context.bot.send_message(chat_id=172778155,
                             parse_mode=telegram.ParseMode.MARKDOWN,
                             text='New request *{}* from *{}*'.format(msg_text, chat))

    RISKY_ASSETS = msg_text.replace(' ', '').strip().split(',')
    # RISKY_ASSETS = ['FB', 'TSLA', 'TWTR', 'MSFT']
    # RISKY_ASSETS = ['GAZP.ME', 'LKOH.ME', 'BANE.ME', 'TATN.ME']

    # Set up the parameters:
    N_PORTFOLIOS = 10 ** 5
    N_DAYS = 252 * N_YEARS
    RISKY_ASSETS.sort()
    START_DATE = datetime.today() - timedelta(days=N_YEARS * 365)
    END_DATE = datetime.today()
    n_assets = len(RISKY_ASSETS)

    # Download the stock prices from Yahoo Finance:
    prices_df = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE, adjusted=True)

    if not prices_df.empty and (prices_df.isnull().values.sum() < .05 * prices_df.size):

        # Calculate annualized average returns and the corresponding standard deviation:
        returns_df = prices_df['Adj Close'].pct_change().dropna()
        avg_returns = returns_df.mean() * N_DAYS
        cov_mat = returns_df.cov() * N_DAYS

        # Simulate random portfolio weights:
        np.random.seed(42)
        weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]

        # Calculate the portfolio metrics:
        portf_rtns = np.dot(weights, avg_returns)
        portf_vol = []
        for i in range(0, len(weights)):
            portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i]))))

        portf_vol = np.array(portf_vol)
        portf_sharpe_ratio = portf_rtns / portf_vol

        # Create a DataFrame containing all the data:
        portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                         'volatility': portf_vol,
                                         'sharpe__ratio': portf_sharpe_ratio})

        # Locate the points creating the Efficient Frontier:
        N_POINTS = 100
        portf_vol_ef = []
        indices_to_skip = []
        portf_rtns_ef = np.linspace(portf_results_df.returns.min(),
                                    portf_results_df.returns.max(),
                                    N_POINTS)
        portf_rtns_ef = np.round(portf_rtns_ef, 2)
        portf_rtns = np.round(portf_rtns, 2)

        for point_index in range(N_POINTS):
            if portf_rtns_ef[point_index] not in portf_rtns:
                indices_to_skip.append(point_index)
                continue
            matched_ind = np.where(portf_rtns ==
                                   portf_rtns_ef[point_index])
            portf_vol_ef.append(np.min(portf_vol[matched_ind]))
        portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)

        # Plot the Efficient Frontier:
        MARKS_ALL = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8",
                     "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
        MARKS = MARKS_ALL[:len(RISKY_ASSETS)]

        fig, ax = plt.subplots()
        portf_results_df.plot(kind='scatter', x='volatility', y='returns', c='sharpe__ratio',
                              cmap='RdYlGn', ax=ax)
        ax.set(xlabel='Volatility',
               ylabel='Expected Returns',
               title='Efficient Frontier')
        ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')

        for asset_index in range(n_assets):
            ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
                       y=avg_returns[asset_index],
                       marker=MARKS[asset_index],
                       s=150,
                       color='black',
                       label=RISKY_ASSETS[asset_index])
        ax.legend()
        plt.savefig('sharp.png')
        plt.close()

        context.bot.send_photo(chat_id=update.effective_chat.id,
                               photo=open('sharp.png', 'rb'),
                               parse_mode=telegram.ParseMode.MARKDOWN)

        max_sharpe_ind = np.argmax(portf_results_df.sharpe__ratio)
        max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]
        min_vol_ind = np.argmin(portf_results_df.volatility)
        min_vol_portf = portf_results_df.loc[min_vol_ind]

        msg_text = ''
        msg_text += '*Maximum Sharpe ratio portfolio*\n\n'
        msg_text += '_Performance_\n'
        for index, value in max_sharpe_portf.items():
            msg_text += f'{index}: {100 * value:.2f}% '
        msg_text += '\n\n_Weights_\n'
        for x, y in zip(RISKY_ASSETS, weights[np.argmax(portf_results_df.sharpe__ratio)]):
            msg_text += f'{x}: {100 * y:.2f}% '

        msg_text += '\n\n*Minimum volatility portfolio*\n\n'
        msg_text += '_Performance_\n'
        for index, value in min_vol_portf.items():
            msg_text += f'{index}: {100 * value:.2f}% '
        msg_text += '\n\n_Weights_\n'
        for x, y in zip(RISKY_ASSETS, weights[np.argmin(portf_results_df.volatility)]):
            msg_text += f'{x}: {100 * y:.2f}% '

        context.bot.send_message(chat_id=update.effective_chat.id,
                                 parse_mode=telegram.ParseMode.MARKDOWN,
                                 text=msg_text)

        # Correlation matrix
        corr_matrix = prices_df[['Adj Close']].corr()

        fig, ax = plt.subplots(figsize=(12, 12))
        ax = sns.heatmap(corr_matrix, annot=True, cmap="BuPu")
        ax.set(xlabel='', ylabel='')
        ax.set_title("Correlation Matrix", fontsize=24)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.savefig('corr.png')
        plt.close()

        context.bot.send_photo(chat_id=update.effective_chat.id,
                               photo=open('corr.png', 'rb'),
                               parse_mode=telegram.ParseMode.MARKDOWN)

    else:
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 parse_mode=telegram.ParseMode.MARKDOWN,
                                 text='Please check correctness of entered tickers.')


def db_write_queries(data_dic):
    summary = Queries(
        user_id=data_dic[0],
        first_name=data_dic[1],
        last_name=data_dic[2],
        username=data_dic[3],
        message=data_dic[4])
    session.add(summary)

    # pushing data to db
    session.commit()
    session.close()


def start(update, context):
    db_write_queries(
        [
            update.effective_user.id,
            update.message.from_user.first_name,
            update.message.from_user.last_name,
            update.message.from_user.username,
            update.message.text])

    context.bot.send_message(chat_id=update.effective_chat.id,
                             parse_mode=telegram.ParseMode.MARKDOWN,
                             text='Just send me list of tickers and i will optimize your Portfolio'
                                  ' for best *Sharpe ratio* or *minimum volatility*.\n'
                                  'Also i will show you *correlation value* between choosen assets.\n\n'
                                  '_Example: AAPL, GAZP.ME, MSFT, YNDX.ME_ \n(Note: russian assets'
                                  ' require a prefix *.ME*)')


def interval(update, context):
    chat = update.effective_chat.id
    msg_text = update.message.text.upper()

    try:
        keyboard = [[InlineKeyboardButton("1 year", callback_data='{}_{}_{}'
                                          .format(chat, msg_text, 1)),
                     InlineKeyboardButton("3 years", callback_data='{}_{}_{}'
                                          .format(chat, msg_text, 3)),
                     InlineKeyboardButton("5 years", callback_data='{}_{}_{}'
                                          .format(chat, msg_text, 5))
                     ]]

        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id=chat, text='Choose historical interval:', reply_markup=reply_markup)
    except:
        bot.send_message(chat_id=chat, text='Too many tickers in one request!')


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


updater.dispatcher.add_handler(CommandHandler('start', start))
text_handler = MessageHandler(Filters.text, interval)
updater.dispatcher.add_handler(text_handler)
updater.dispatcher.add_handler(CallbackQueryHandler(sharp))

# log all errors
updater.dispatcher.add_error_handler(error)

updater.start_polling()
updater.idle()
