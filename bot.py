# -*- coding: utf-8 -*-
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
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
session = get_db_session(expire_on_commit=False)
updater = Updater(token, use_context=True)
logging.basicConfig(filename='logs.log', level=logging.WARNING)
current_path = os.path.dirname(os.path.realpath(__file__))


def matrix(corr_matrix, update, context):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax = sns.heatmap(corr_matrix, annot=True, cmap="BuPu")
    ax.set(xlabel='', ylabel='')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig('corr.png')
    plt.close()

    context.bot.send_message(chat_id=update.effective_chat.id,
                             parse_mode=telegram.ParseMode.MARKDOWN,
                             text='*Матрица корреляций для портфеля*')
    context.bot.send_photo(chat_id=update.effective_chat.id,
                           photo=open('corr.png', 'rb'),
                           parse_mode=telegram.ParseMode.MARKDOWN)

    context.bot.send_message(chat_id=update.effective_chat.id,
                             parse_mode=telegram.ParseMode.MARKDOWN,
                             text='🔶 Значение *«1»* - это сильная *прямая корреляция*, говорящая нам о том, '
                                  'что на заданном '
                                  'временном интервале'
                                  ' рост/падением одного актива сопровождается ростом/падением второго.\n'
                                  'В контексте фондового рынка это очень рисковано и как раз воспроизводит ситуацию '
                                  '«хранения яиц в одной корзине».\n\n'
                                  '🔶 Значение *«-1»* - это сильная *обратная корреляция*, говорящая нам о том, '
                                  'что на заданном '
                                  'временном интервале'
                                  ' рост одного актива сопровождается падением второго и наоборот.\n'
                                  'Это тот самый случай к которому нам необходимо стремиться, следуя идее '
                                  '*диверсификации.*')


def sharp(ticker, cov_matrix, mean_returns, update, context):
    # создаем массив из нулей
    num_iterations = 10000
    simulation_res = np.zeros((4 + len(ticker) - 1, num_iterations))

    # сама итерация
    for i in range(num_iterations):
        # Выбрать случайные веса и нормализовать, чтоб сумма равнялась 1
        weights = np.array(np.random.random(len(ticker)))
        weights /= np.sum(weights)

        # Вычислить доходность и стандартное отклонение
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Сохранить все полученные значения в массив
        simulation_res[0, i] = portfolio_return
        simulation_res[1, i] = portfolio_std_dev

        # Вычислить коэффициент Шарпа и сохранить
        simulation_res[2, i] = simulation_res[0, i] / simulation_res[1, i]

        # Сохранить веса
        for j in range(len(weights)):
            simulation_res[j + 3, i] = weights[j]

    # сохраняем полученный массив в DataFrame для построения данных и анализа.
    sim_frame = pd.DataFrame(simulation_res.T, columns=['ret', 'stdev', 'sharp'] + ticker)

    # узнать максимальный sharp Ratio
    max_sharp = sim_frame.iloc[sim_frame['sharp'].idxmax()]

    # узнать минимальное стандартное отклонение
    min_std = sim_frame.iloc[sim_frame['stdev'].idxmin()]

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(sim_frame.stdev, sim_frame.ret, c=sim_frame.sharp, cmap='RdYlBu')
    plt.xlabel('Волатильность')
    plt.ylabel('Доходность')

    plt.scatter(max_sharp[1], max_sharp[0], marker="o", color='r', s=300)
    plt.scatter(min_std[1], min_std[0], marker="o", color='b', s=300)
    plt.savefig('sharp.png')
    plt.close()

    # собираем данные для текстового вывода
    best_sharp_msg = ''
    best_std_msg = ''
    for i in ticker:
        best_sharp_msg += '{}: {}%\n'.format(i, round(100*max_sharp[i]))
        best_std_msg += '{}: {}%\n'.format(i, round(100*min_std[i]))

    context.bot.send_message(chat_id=update.effective_chat.id,
                             parse_mode=telegram.ParseMode.MARKDOWN,
                             text='*Оптимальные соотношения акций в портфеле*')

    context.bot.send_photo(chat_id=update.effective_chat.id,
                           photo=open('sharp.png', 'rb'),
                           parse_mode=telegram.ParseMode.MARKDOWN)

    context.bot.send_message(chat_id=update.effective_chat.id,
                             parse_mode=telegram.ParseMode.MARKDOWN,
                             text='🔴 - _Лучшее значение коэффициента Шарпа_\n'
                                  '*Доходность:* {}\n'
                                  '*Волатильность:* {}\n'
                                  '*Коэффициент Шарпа:* {}\n'
                                  '*Портфель:*\n{}\n\n'
                                  '🔵 - _Самый низкий риск_\n'
                                  '*Доходность:* {}\n'
                                  '*Волатильность:* {}\n'
                                  '*Коэффициент Шарпа:* {}\n'
                                  '*Портфель:*\n{}\n\n'
                                  ''.format(round(max_sharp['ret'], 5),
                                            round(max_sharp['stdev'], 5),
                                            round(max_sharp['sharp'], 5),
                                            best_sharp_msg,
                                            round(min_std['ret'], 5),
                                            round(min_std['stdev'], 5),
                                            round(min_std['sharp'], 5),
                                            best_std_msg))


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
                             text='Привет, я небольшой телеграмм бот, который пытается помочь'
                                  ' начинающим инвесторам быть более осмысленными.\n\n'
                                  '🔷 Можно долго спорить об эффективности торговых стратегий, '
                                  'но моё субъективное видение заключается в том, что наиболее '
                                  'разумно и безопасно следовать идее *портфельного инвестирования*, '
                                  'с обязательной *диверсификацией вложений*.\n\n'
                                  '🔷 Простым языком это значит, что «не нужно держать все яйца '
                                  'в одной корзине».\n'
                                  'А при покупке активов руководствоваться выбором компаний из разных стран '
                                  'и секторов экономики.\n\n'
                                  '🔷 Ключевым в данной стратегии является проверка *степени корреляции* выбранных '
                                  'активов.\n'
                                  'Звучит страшно, но идея очень проста.\n'
                                  'Необходимо выбирать активы '
                                  'таким образом, чтобы их *корреляция* была минимальна, '
                                  'а в идеале - «отрицательна».\n\n'
                                  '🔷 Что же такое *корреляция*?\n'
                                  '*Корреляция* - это взаимосвязь двух случайных величин'
                                  ' (в нашем случае цен), показывающая как сильно изменение одной из величин,'
                                  ' «сопровождается» изменением второй. *Коэффициент корреляции* принимает значения '
                                  '*от -1 '
                                  ' до 1*, и показывает силу этой связи.\n\n'
                                  '🔶 Значение *«1»* - это сильная *прямая корреляция*, говорящая нам о том, '
                                  'что на заданном '
                                  'временном интервале'
                                  ' рост/падением одного актива сопровождается ростом/падением второго.\n'
                                  'В контексте фондового рынка это очень рисковано и как раз воспроизводит ситуацию '
                                  '«хранения яиц в одной корзине».\n\n'
                                  '🔶 Значение *«-1»* - это сильная *обратная корреляция*, говорящая нам о том, '
                                  'что на заданном '
                                  'временном интервале'
                                  ' рост одного актива сопровождается падением второго и наоборот.\n'
                                  'Это тот самый случай к которому нам необходимо стремиться, следуя идее '
                                  '*диверсификации.*')

    context.bot.send_message(chat_id=update.effective_chat.id,
                             parse_mode=telegram.ParseMode.MARKDOWN,
                             text='«Вишенкой на торте» данной стратегии является идея,'
                                  ' предложенная нобелевским лауреатом *Ульямом Шарпом*.\n\n'
                                  '🔷 Им было показано, что при «правильном» соотношении активов'
                                  ' в портфеле, одна и та же *доходность* (см. вики) может быть'
                                  ' достигнута при разной *степени риска* (волатильности).\n\n'
                                  '🔷 Для выбора этого наиболее эффективного соотношения принято '
                                  'использовать одноимённый *коэффициент Шарпа*. Он показывает '
                                  'избыточность доходности портфеля на единицу риска (см. вики)\n\n'
                                  'На мой взгляд, два описанных выше шага, уже дадут неплохое'
                                  ' подспорье к «осмысленным» инвестициям, а я Вам в этом помогу)\n\n'
                                  'Отправьте мне интересующие тикеры (см. вики) компаний, а я'
                                  ' построю для Вас матрицу корреляции и распределение доходности от рисков.\n\n'
                                  '*Для российских компаний, представленным на МосБирже необходимо'
                                  ' добавить префикс «.ME»*\n'
                                  '_Например: AAPL, GAZP.ME, FIVE.ME, AMZN_')


def sharp_n_matrix(update, context):
    msg_text = update.message.text.upper()
    try:
        db_write_queries(
            [
                update.effective_user.id,
                update.message.from_user.first_name,
                update.message.from_user.last_name,
                update.message.from_user.username,
                update.message.text])

        ticker = msg_text.replace(' ', '').strip().split(',')
        stock = yf.download(ticker, datetime.today() - timedelta(days=365), datetime.today())

        all_adj_close = stock[['Adj Close']]

        if not all_adj_close.empty and (all_adj_close.isnull().values.sum() < .5 * all_adj_close.size):
            # Ежедневная доходность
            all_returns = all_adj_close.pct_change()

            # Матрицы корреляции и ковариации
            corr_matrix = all_returns.corr()
            cov_matrix = all_returns.corr()

            # Средняя доходность за день
            all_returns_mean = all_returns.mean()

            # построения графиков
            sharp(ticker, cov_matrix, all_returns_mean, update, context)
            time.sleep(3)
            matrix(corr_matrix, update, context)
        else:
            context.bot.send_message(chat_id=update.effective_chat.id,
                                     parse_mode=telegram.ParseMode.MARKDOWN,
                                     text='Данные по одной из компаний списка отсутствуют,'
                                          ' либо Вы ошиблись в формате ввода.\n'
                                          'Введите тикеры интересующих Вас компаний через запятую\n\n'
                                          '_Например: AAPL, GAZP.ME, AMZN_')
    except:
        logging.exception("Error")
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 parse_mode=telegram.ParseMode.MARKDOWN,
                                 text='Данные по одной из компаний списка отсутствуют,'
                                      ' либо Вы ошиблись в формате ввода.\n'
                                      'Введите тикеры интересующих Вас компаний через запятую\n\n'
                                      '_Например: AAPL, GAZP.ME, AMZN_')


updater.dispatcher.add_handler(CommandHandler('start', start))
text_handler = MessageHandler(Filters.text, sharp_n_matrix)
updater.dispatcher.add_handler(text_handler)

updater.start_polling()
updater.idle()
