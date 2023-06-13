{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "791eb536",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "%matplotlib inline\n",
    "import matplotlib \n",
    "matplotlib.rcParams[\"figure.figsize\"] = (20,10)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "70fe7305",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>area_type</th>\n",
       "      <th>availability</th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>society</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>balcony</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>19-Dec</td>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>Coomee</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>39.07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Plot  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>Theanmp</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>120.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>62.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>Soiewre</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>95.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Super built-up  Area</td>\n",
       "      <td>Ready To Move</td>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>51.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              area_type   availability                  location       size  \\\n",
       "0  Super built-up  Area         19-Dec  Electronic City Phase II      2 BHK   \n",
       "1            Plot  Area  Ready To Move          Chikka Tirupathi  4 Bedroom   \n",
       "2        Built-up  Area  Ready To Move               Uttarahalli      3 BHK   \n",
       "3  Super built-up  Area  Ready To Move        Lingadheeranahalli      3 BHK   \n",
       "4  Super built-up  Area  Ready To Move                  Kothanur      2 BHK   \n",
       "\n",
       "   society total_sqft  bath  balcony   price  \n",
       "0  Coomee        1056   2.0      1.0   39.07  \n",
       "1  Theanmp       2600   5.0      3.0  120.00  \n",
       "2      NaN       1440   2.0      3.0   62.00  \n",
       "3  Soiewre       1521   3.0      1.0   95.00  \n",
       "4      NaN       1200   2.0      1.0   51.00  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1 = pd.read_csv(\"bengaluru_house_prices.csv\")\n",
    "df1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "52491900",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13320, 9)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d39422d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['area_type', 'availability', 'location', 'size', 'society',\n",
       "       'total_sqft', 'bath', 'balcony', 'price'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df1.columns\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "001f69f5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Super built-up  Area', 'Plot  Area', 'Built-up  Area',\n",
       "       'Carpet  Area'], dtype=object)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1['area_type'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b7491e43",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "area_type\n",
       "Built-up  Area          2418\n",
       "Carpet  Area              87\n",
       "Plot  Area              2025\n",
       "Super built-up  Area    8790\n",
       "Name: area_type, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1.groupby('area_type')['area_type'].agg('count')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "2e1223cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size total_sqft  bath   price\n",
       "0  Electronic City Phase II      2 BHK       1056   2.0   39.07\n",
       "1          Chikka Tirupathi  4 Bedroom       2600   5.0  120.00\n",
       "2               Uttarahalli      3 BHK       1440   2.0   62.00\n",
       "3        Lingadheeranahalli      3 BHK       1521   3.0   95.00\n",
       "4                  Kothanur      2 BHK       1200   2.0   51.00"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')\n",
    "df2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "31d8b877",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location       1\n",
       "size          16\n",
       "total_sqft     0\n",
       "bath          73\n",
       "price          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c160eec7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location      0\n",
       "size          0\n",
       "total_sqft    0\n",
       "bath          0\n",
       "price         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "df3 = df2.dropna()\n",
    "df3.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "59eebce2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13246, 5)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "10637128",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['2 BHK', '4 Bedroom', '3 BHK', '4 BHK', '6 Bedroom', '3 Bedroom',\n",
       "       '1 BHK', '1 RK', '1 Bedroom', '8 Bedroom', '2 Bedroom',\n",
       "       '7 Bedroom', '5 BHK', '7 BHK', '6 BHK', '5 Bedroom', '11 BHK',\n",
       "       '9 BHK', '9 Bedroom', '27 BHK', '10 Bedroom', '11 Bedroom',\n",
       "       '10 BHK', '19 BHK', '16 BHK', '43 Bedroom', '14 BHK', '8 BHK',\n",
       "       '12 Bedroom', '13 BHK', '18 Bedroom'], dtype=object)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3['size'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "51056626",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\brnpr\\AppData\\Local\\Temp\\ipykernel_3416\\3603722699.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df3['bhk']= df3['size'].apply(lambda x: int(x.split(' ')[0]))\n"
     ]
    }
   ],
   "source": [
    "df3['bhk']= df3['size'].apply(lambda x: int(x.split(' ')[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "36ef8e53",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size total_sqft  bath   price  bhk\n",
       "0  Electronic City Phase II      2 BHK       1056   2.0   39.07    2\n",
       "1          Chikka Tirupathi  4 Bedroom       2600   5.0  120.00    4\n",
       "2               Uttarahalli      3 BHK       1440   2.0   62.00    3\n",
       "3        Lingadheeranahalli      3 BHK       1521   3.0   95.00    3\n",
       "4                  Kothanur      2 BHK       1200   2.0   51.00    2"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "941270cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 2,  4,  3,  6,  1,  8,  7,  5, 11,  9, 27, 10, 19, 16, 43, 14, 12,\n",
       "       13, 18], dtype=int64)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3['bhk'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "cf75f0ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1718</th>\n",
       "      <td>2Electronic City Phase II</td>\n",
       "      <td>27 BHK</td>\n",
       "      <td>8000</td>\n",
       "      <td>27.0</td>\n",
       "      <td>230.0</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4684</th>\n",
       "      <td>Munnekollal</td>\n",
       "      <td>43 Bedroom</td>\n",
       "      <td>2400</td>\n",
       "      <td>40.0</td>\n",
       "      <td>660.0</td>\n",
       "      <td>43</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       location        size total_sqft  bath  price  bhk\n",
       "1718  2Electronic City Phase II      27 BHK       8000  27.0  230.0   27\n",
       "4684                Munnekollal  43 Bedroom       2400  40.0  660.0   43"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3[df3.bhk>20]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b48b8534",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['1056', '2600', '1440', ..., '1133 - 1384', '774', '4689'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3.total_sqft.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a1af276e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def is_float(x):\n",
    "    try:\n",
    "        float(x)\n",
    "    except:\n",
    "        return False\n",
    "    return True\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "85c95ade",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "is_float(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "dce58540",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>Yelahanka</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>2100 - 2850</td>\n",
       "      <td>4.0</td>\n",
       "      <td>186.000</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>122</th>\n",
       "      <td>Hebbal</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>3067 - 8156</td>\n",
       "      <td>4.0</td>\n",
       "      <td>477.000</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>137</th>\n",
       "      <td>8th Phase JP Nagar</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1042 - 1105</td>\n",
       "      <td>2.0</td>\n",
       "      <td>54.005</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>165</th>\n",
       "      <td>Sarjapur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1145 - 1340</td>\n",
       "      <td>2.0</td>\n",
       "      <td>43.490</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>188</th>\n",
       "      <td>KR Puram</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1015 - 1540</td>\n",
       "      <td>2.0</td>\n",
       "      <td>56.800</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               location   size   total_sqft  bath    price  bhk\n",
       "30            Yelahanka  4 BHK  2100 - 2850   4.0  186.000    4\n",
       "122              Hebbal  4 BHK  3067 - 8156   4.0  477.000    4\n",
       "137  8th Phase JP Nagar  2 BHK  1042 - 1105   2.0   54.005    2\n",
       "165            Sarjapur  2 BHK  1145 - 1340   2.0   43.490    2\n",
       "188            KR Puram  2 BHK  1015 - 1540   2.0   56.800    2"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3[~df3['total_sqft'].apply(is_float)].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "73448855",
   "metadata": {},
   "outputs": [],
   "source": [
    "def convert_sqft_to_num(x):\n",
    "    tokens = x.split('-')\n",
    "    if len(tokens) == 2:\n",
    "        return (float(tokens[0])+float(tokens[1]))/2\n",
    "    try:\n",
    "        return float(x)\n",
    "    except:\n",
    "        return None   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "25a60ccd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2166.0"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "convert_sqft_to_num('2166')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "77dbd897",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2475.0"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "convert_sqft_to_num('2100 - 2850')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "a15c0285",
   "metadata": {},
   "outputs": [],
   "source": [
    "convert_sqft_to_num('34.45sq. Meter')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "71e3542c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  bhk\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2\n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df4 = df3.copy()\n",
    "df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)\n",
    "df4 = df4[df4.total_sqft.notnull()]\n",
    "df4.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "8c66442f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location      Yelahanka\n",
       "size              4 BHK\n",
       "total_sqft       2475.0\n",
       "bath                4.0\n",
       "price             186.0\n",
       "bhk                   4\n",
       "Name: 30, dtype: object"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df4.loc[30]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "b4b32ac6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "      <td>3699.810606</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "      <td>4615.384615</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "      <td>4305.555556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "      <td>6245.890861</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "      <td>4250.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  bhk  \\\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2   \n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4   \n",
       "2               Uttarahalli      3 BHK      1440.0   2.0   62.00    3   \n",
       "3        Lingadheeranahalli      3 BHK      1521.0   3.0   95.00    3   \n",
       "4                  Kothanur      2 BHK      1200.0   2.0   51.00    2   \n",
       "\n",
       "   price_per_sqft  \n",
       "0     3699.810606  \n",
       "1     4615.384615  \n",
       "2     4305.555556  \n",
       "3     6245.890861  \n",
       "4     4250.000000  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5 = df4.copy()\n",
    "df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']\n",
    "df5.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "e46ec168",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1298"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df5.location.unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "a72c89c1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location\n",
       "Whitefield               533\n",
       "Sarjapur  Road           392\n",
       "Electronic City          304\n",
       "Kanakpura Road           264\n",
       "Thanisandra              235\n",
       "                        ... \n",
       "1 Giri Nagar               1\n",
       "Kanakapura Road,           1\n",
       "Kanakapura main  Road      1\n",
       "Kannur                     1\n",
       "whitefiled                 1\n",
       "Name: location, Length: 1287, dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5.location = df5.location.apply(lambda x: x.strip())\n",
    "location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)\n",
    "location_stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "c1ad6b9f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1047"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "len(location_stats[location_stats<=10])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "9c91bb8b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "location\n",
       "Sadashiva Nagar          10\n",
       "Naganathapura            10\n",
       "Basapura                 10\n",
       "Nagadevanahalli          10\n",
       "Kalkere                  10\n",
       "                         ..\n",
       "1 Giri Nagar              1\n",
       "Kanakapura Road,          1\n",
       "Kanakapura main  Road     1\n",
       "Kannur                    1\n",
       "whitefiled                1\n",
       "Name: location, Length: 1047, dtype: int64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "location_stats_less_than_10 = location_stats[location_stats<=10]\n",
    "location_stats_less_than_10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "6f77d1eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1287"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "len(df5.location.unique())\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "bfc4bdac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "241"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)\n",
    "len(df5.location.unique())\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "1be00e5d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Electronic City Phase II</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1056.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>39.07</td>\n",
       "      <td>2</td>\n",
       "      <td>3699.810606</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Chikka Tirupathi</td>\n",
       "      <td>4 Bedroom</td>\n",
       "      <td>2600.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>4</td>\n",
       "      <td>4615.384615</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Uttarahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1440.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>62.00</td>\n",
       "      <td>3</td>\n",
       "      <td>4305.555556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lingadheeranahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1521.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>95.00</td>\n",
       "      <td>3</td>\n",
       "      <td>6245.890861</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Kothanur</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>51.00</td>\n",
       "      <td>2</td>\n",
       "      <td>4250.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Whitefield</td>\n",
       "      <td>2 BHK</td>\n",
       "      <td>1170.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>38.00</td>\n",
       "      <td>2</td>\n",
       "      <td>3247.863248</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Old Airport Road</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>2732.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>204.00</td>\n",
       "      <td>4</td>\n",
       "      <td>7467.057101</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Rajaji Nagar</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>3300.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>600.00</td>\n",
       "      <td>4</td>\n",
       "      <td>18181.818182</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Marathahalli</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1310.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>63.25</td>\n",
       "      <td>3</td>\n",
       "      <td>4828.244275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>other</td>\n",
       "      <td>6 Bedroom</td>\n",
       "      <td>1020.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>370.00</td>\n",
       "      <td>6</td>\n",
       "      <td>36274.509804</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   location       size  total_sqft  bath   price  bhk  \\\n",
       "0  Electronic City Phase II      2 BHK      1056.0   2.0   39.07    2   \n",
       "1          Chikka Tirupathi  4 Bedroom      2600.0   5.0  120.00    4   \n",
       "2               Uttarahalli      3 BHK      1440.0   2.0   62.00    3   \n",
       "3        Lingadheeranahalli      3 BHK      1521.0   3.0   95.00    3   \n",
       "4                  Kothanur      2 BHK      1200.0   2.0   51.00    2   \n",
       "5                Whitefield      2 BHK      1170.0   2.0   38.00    2   \n",
       "6          Old Airport Road      4 BHK      2732.0   4.0  204.00    4   \n",
       "7              Rajaji Nagar      4 BHK      3300.0   4.0  600.00    4   \n",
       "8              Marathahalli      3 BHK      1310.0   3.0   63.25    3   \n",
       "9                     other  6 Bedroom      1020.0   6.0  370.00    6   \n",
       "\n",
       "   price_per_sqft  \n",
       "0     3699.810606  \n",
       "1     4615.384615  \n",
       "2     4305.555556  \n",
       "3     6245.890861  \n",
       "4     4250.000000  \n",
       "5     3247.863248  \n",
       "6     7467.057101  \n",
       "7    18181.818182  \n",
       "8     4828.244275  \n",
       "9    36274.509804  "
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df5.head(10)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "910d8c7b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13200, 7)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df5.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "2af7bb96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12456, 7)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df6 = df5[~(df5.total_sqft/df5.bhk<300)]\n",
    "df6.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "299e8c7b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     12456.000000\n",
       "mean       6308.502826\n",
       "std        4168.127339\n",
       "min         267.829813\n",
       "25%        4210.526316\n",
       "50%        5294.117647\n",
       "75%        6916.666667\n",
       "max      176470.588235\n",
       "Name: price_per_sqft, dtype: float64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df6.price_per_sqft.describe()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "119ce54a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10242, 7)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "def remove_pps_outliers(df):\n",
    "    df_out = pd.DataFrame()\n",
    "    for key, subdf in df.groupby('location'):\n",
    "        m = np.mean(subdf.price_per_sqft)\n",
    "        st = np.std(subdf.price_per_sqft)\n",
    "        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]\n",
    "        df_out = pd.concat([df_out,reduced_df],ignore_index=True)\n",
    "    return df_out\n",
    "df7 = remove_pps_outliers(df6)\n",
    "df7.shape\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "5146a8c2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3sAAAJcCAYAAABAE73ZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA+G0lEQVR4nO3df5hkZ1kn/O89SUxkpjGBBBgnYIIZJCHqiE3EdVY7KAJ5EURxJy6uxGRXlx8KRBBQd4366psFEZVdZYP6BgHNRFGJbMKvwAxvAMlOIBBIxBklY0IihJ/2zJqQZJ73j6rOFDPdPTUzXV1dpz+f66qrqp9zququPnSmvtzPeU611gIAAEC3rBl3AQAAACw9YQ8AAKCDhD0AAIAOEvYAAAA6SNgDAADoIGEPAACgg4Q9AFaNqnpOVb1ryH1/sar+sP/4UVW1p6qOGW2FALB0ynX2AJgkVXVrkocnuT/JniTvSPLC1tqeMdf09Uke3Vrb2x/7j0l+orU2M666AFjddPYAmEQ/1Fpbl2RTku9I8srxlpMkOTbJi8ZdRFUdO+4aAFgZhD0AJlZr7Z+TvDO90JckqapXVNU/VNVsVd1cVc8a2HZBVV038PPvVtVtVfUvVXVDVf3bgW2XVNWb+49Pq6p2iCD16iQvraoT59t4iPf6+qp6Y1V9qapuqapfqKrbD+MzfaCqXltVX0xyyTC/OwC6T9gDYGJV1alJnpZk18DwPyT5t0m+IcmvJnlzVa1f4CX+d3pB8SFJ/jTJn1fVCUdYzo4k25K89Aje61eSnJbk0UmenOQnDnjuoT7TdyX5xyQPS/IbR1g/AB0j7AEwif66qmaT3Jbkc+mFpSRJa+3PW2t3tNb2tda2JtmZ5Jz5XqS19ubW2hdaa/e11l6T5Pgk33IUdf3XJD9bVacc5nv9uyS/2Vr7Umvt9iS/d8BzD/WZ7mitva7/2v96FPUD0CHCHgCT6Idba1NJZpI8NsnJcxuq6ier6saq+nJVfTnJ2YPbB1XVz/enTX6lv+83LLTvMFprn0jy9iSvOMz3+sb0guuc2w547qE+09fsDwCJsAfABGutbU9yeZLfSpKq+qYkb0jywiQPba2dmOQTSerA5/bPmXt5el21k/r7fmW+fQ/TryT5T0k2HMZ73Znk1IHXeOTAc4f5TJbWBuAgwh4Ak+53kjy5qjYlWZte8LkrSarqp9Lrgs1nKsl9/X2Prar/muTBR1tMa21Xkq1Jfu4w3uvKJK+sqpOqakN6wW7O4XwmAHiAsAfARGut3ZXkT5L8l9bazUlek+RDST6b5FuTfGCBp74zyTVJ/j7J7iR3Z+mmQ/5aeiFt2Pf6tSS3J/l0kvck+Ysk9yTJYX4mAHiAi6oDsGpU1YXpXej8SUPs+2tJTm2tXTj6yg567+clOb+19n3L/d4AdIfOHgCryePS654tqqoqyVnD7LsUqmp9VX1PVa2pqm9J8vNJ/mo53huA7lrs4rAA0BlV9ddJNib5sSF2/0h60yhfeKgdl8jXJfmfSU5P8uUkVyT5/WV6bwA6yjROAACADjKNEwAAoIMmehrnySef3E477bRxlwEAADAWN9xww+dba6fMt22iw95pp52WHTt2jLsMAACAsaiq3QttM40TAACgg4Q9AACADhL2AAAAOmiiz9mbz7333pvbb789d99997hLGasTTjghp556ao477rhxlwIAAIxB58Le7bffnqmpqZx22mmpqnGXMxattXzhC1/I7bffntNPP33c5QAAAGPQuWmcd999dx760Ieu2qCXJFWVhz70oau+uwkAAKtZ58JeklUd9Ob4HQAAwOrWybAHAACw2gl7S+y2227LueeemzPPPDOPe9zj8ru/+7vz7nfJJZdkw4YN2bRpUx772Mfmec97Xvbt25ckueCCC/IXf/EXX7P/unXrkiS33nprzj777AfG3/CGN+Txj398vvSlL43oEwEAAJNo1Ye92dnkD/8wefnLe/ezs0f3escee2xe85rX5JZbbsnf/u3f5n/8j/+Rm2++ed59X/KSl+TGG2/MzTffnJtuuinbt28/rPd605velNe97nV517velZNOOunoCgcAADqlc6txHo7rrkvOOy/Zty/ZuzdZuza5+OLk6quTzZuP7DXXr1+f9evXJ0mmpqZy5pln5jOf+UzOOuusBZ/z1a9+NXffffdhBbYrr7wyl156aa699tqcfPLJR1YsAADQWau2szc72wt6s7O9oJf07ufG9+w5+ve49dZb89GPfjTf9V3fNe/21772tdm0aVPWr1+fxzzmMdm0adMD2172spdl06ZND9wG7d69Oy984Qvzrne9K494xCOOvlAAAKBzVm3Y27q119Gbz759ve1HY8+ePfnRH/3R/M7v/E4e/OAHz7vP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\n",
      "text/plain": [
       "<Figure size 1080x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "\n",
    "def plot_scatter_chart(df,location):\n",
    "    bhk2 = df[(df.location==location) & (df.bhk==2)]\n",
    "    bhk3 = df[(df.location==location) & (df.bhk==3)]\n",
    "    matplotlib.rcParams['figure.figsize'] = (15,10)\n",
    "    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)\n",
    "    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)\n",
    "    plt.xlabel(\"Total Square Feet Area\")\n",
    "    plt.ylabel(\"Price (Lakh Indian Rupees)\")\n",
    "    plt.title(location)\n",
    "    plt.legend()\n",
    "    \n",
    "plot_scatter_chart(df7,\"Rajaji Nagar\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "d32639be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3sAAAJcCAYAAABAE73ZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA//UlEQVR4nO3dfZikZ10n+u9vkpjAzGgCCSZMwEQyLIQIIzSI6xztiBLIhSKCTljWJUs8sLysQIQD7nqWqIubAyoqvi2gOzGizAi4RJYgrzNsFMlOMPKS6E7WJBISSQgJdgaSTDL3+aOqM5WZ7p7qma6u7qc+n+vqq6ru56mqX3UqDd/87ue+q7UWAAAAumXNuAsAAABg6Ql7AAAAHSTsAQAAdJCwBwAA0EHCHgAAQAcJewAAAB0k7AHAkKrqoqr6owWO31BVP3SYr72jqn768KsDgAcT9gCYKHMFsqo6v6quGFdNADAKwh4AAEAHCXsAMKCqHllV76uq26rq+qr6mQNOOa6qtlXVTFV9tqqedMDxp1bVNVV1R1X9t6o6rv+6J1TVB/uve0f//qnL86kAmETCHgD0VdWaJH+e5G+TbEjyjCSvqapzBk57bpI/TfKwJH+c5L9X1TEDx1+U5Jwkj0ny2CQ/3x9fk+S/JfmOJI9O8s0kvzWyDwPAxBP2AJhE/72q7pz9SfI7/fGnJjmptfaLrbV7W2v/kOSdSc4beO5VrbX3ttb2Jvm1JMclefrA8d9qrX2ptfa1JG9O8sIkaa3d3lp7X2vtG621mf6xHxjppwRgoh097gIAYAx+rLX2sdkHVXV+kp9Or+v2yH4AnHVUkv858PhLs3daa/uq6qYkj5zreJIbZ49V1UOTvC3Js5Kc0D++vqqOaq3df6QfCAAOJOwBwH5fSnJ9a23jAuc8avZOf9rnqUlunut4etM1Z4/9bJJ/keR7Wmv/VFWbkvxNklqCugHgIKZxAsB+Vyb556p6Q1U9pKqOqqqzquqpA+c8pap+vKqOTvKaJPck+euB46+sqlOr6mFJ/kOSbf3x9eldp3dn/9ibRv5pAJhowh4A9PWnU/5Ikk1Jrk/y1STvSvJtA6d9IMmWJHck+akkP96/fm/WHyf5SJJ/6P/85/74ryd5SP81/zrJh0f0MQAgSVKttXHXAAAAwBLT2QMAAOggYQ8AAKCDhD0AAIAOEvYAAAA6aFXvs3fiiSe20047bdxlAAAAjMVVV1311dbaSXMdW9Vh77TTTsuuXbvGXQYAAMBYVNWN8x0zjRMAAKCDhD0AAIAOEvYAAAA6aFVfszeXvXv35qabbsrdd9897lLG6rjjjsupp56aY445ZtylAAAAY9C5sHfTTTdl/fr1Oe2001JV4y5nLFpruf3223PTTTfl9NNPH3c5AADAGHRuGufdd9+dhz/84RMb9JKkqvLwhz984rubAAAwyToX9pJMdNCb5XcAAACTrZNhDwAAYNIJe0vsS1/6Us4+++w8/vGPzxOe8IT8xm/8xpznXXTRRdmwYUM2bdqUxz3ucXn5y1+effv2JUnOP//8vPe9733Q+evWrUuS3HDDDTnrrLMeGH/nO9+ZJz/5ybnjjjtG9IkAAIDVaOLD3sxM8q53JW94Q+92ZubIXu/oo4/Or/7qr+baa6/NX//1X+e3f/u3c80118x57mtf+9pcffXVueaaa/L5z38+O3fuXNR7XXrppXn729+ej3zkIznhhBOOrHAAAKBTOrca52JccUVy7rnJvn3Jnj3J2rXJhRcmH/pQsnnz4b3mKaecklNOOSVJsn79+jz+8Y/Pl7/85Zx55pnzPufee+/N3XffvajAtn379lx88cX5+Mc/nhNPPPHwigUAADprYjt7MzO9oDcz0wt6Se92dvyuu478PW644Yb8zd/8Tb7ne75nzuNve9vbsmnTppxyyil57GMfm02bNj1w7PWvf302bdr0wM+gG2+8Ma961avykY98JCeffPKRFwoAAHTOxIa9bdt6Hb257NvXO34k7rrrrjz/+c/Pr//6r+dbv/Vb5zxndhrnrbfemj179uQ973nPA8fe+ta35uqrr37gZ9BJJ52URz/60dm+ffuRFQkAAHTWxIa93bv3d/QOtGdPct11h//ae/fuzfOf//y86EUvyo//+I8f8vxjjjkmz3rWs/KpT31qqNd/6EMfmssvvzy/93u/l3e/+92HXygAANBZE3vN3saNvWv05gp8a9cmZ5xxeK/bWssFF1yQxz/+8bnwwguHfs5f/dVfHTRdcyEnnXRSPvzhD2d6ejonnnhizjnnnMMrGAAA6KSJ7ext2ZKsmefTr1nTO344/vIv/zKXXnppPvGJTzxwvd2HPvShOc+dvWbvrLPOyn333ZdXvOIVi3qv008/PZdddlle8pKX5DOf+czhFQwAAHRStdbGXcNhm5qaart27XrQ2LXXXpvHP/7xQz1/rtU416w5stU4V5LF/C4AAIDVp6quaq1NzXVsYqdxJr1Ad/PNvcVYrruuN3Vzy5akv385AADAqjXRYS/pBbsLLhh3FQAAAEtrYq/ZAwAAGMb01ulMb50edxmLJuwBAAB0kLAHAADQQRN/zR4AAMCBBqdt7rxx50FjO87fsbwFHQadvSV2991352lPe1qe9KQn5QlPeELe9KY3zXneRRddlA0bNmTTpk153OMel5e//OXZt29fkuT888/Pe9/73gedv66/ROgNN9yQs84664Hxd77znXnyk5+cO+64Y0SfCAAAWI109rI/oS9FOj/22GPziU98IuvWrcvevXuzefPmPPvZz87Tn/70g8597Wtfm9e97nXZt29fvv/7vz87d+7M2WefPfR7XXrppXn729+eT3ziEznhhBOOuHYAAKBnMBssZV5YTsLeEquqB7pwe/fuzd69e1NVCz7n3nvvzd13372owLZ9+/ZcfPHF+fjHP54TTzzxiGoGAAC6xzTOEbj//vuzadOmPOIRj8gP//AP53u+53vmPO9tb3tbNm3alFNOOSWPfexjs2nTpgeOvf71r8+mTZse+Bl044035lWvelU+8pGP5OSTTx7hJwEAAFarie3sjfKCy6OOOipXX3117rzzzjzvec/LF77whQddZzdrdhrn3r1784IXvCDvec97ct555yVJ3vrWt+YFL3jBA+fOdguT5KSTTsrDHvawbN++Pa997WsPu04AAODQVtv0zVk6eyN0/PHHZ3p6Oh/+8IcXPO+YY47Js571rHzqU58a6nUf+tCH5vLLL8/v/d7v5d3vfvdSlAoAAHTMxHb2RnXB5W233ZZjjjkmxx9/fL75zW/mYx/7WN7whjcs+JzWWv7qr/7qoOmaCznppJPy4Q9/ONPT0znxxBNzzjnnHGHlAABAl+jsLbFbbrklZ599dp74xCfmqU99an74h384z3nOc+Y8d/aavbPOOiv33XdfXvGKVyzqvU4//fRcdtlleclLXpLPfOYzS1E+AADQEdVaG3cNh21qaqrt2rXrQWPXXnttHv/4xy/qdVbrUqqHcji/CwAAYPWoqqtaa1NzHZvYaZyDuhbyAAAATOMEAADooE6GvdU8NXWp+B0AAMBk61zYO+6443L77bdP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\n",
      "text/plain": [
       "<Figure size 1080x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "\n",
    "plot_scatter_chart(df7,\"Hebbal\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "db3039f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7317, 7)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def remove_bhk_outliers(df):\n",
    "    exclude_indices = np.array([])\n",
    "    for location, location_df in df.groupby('location'):\n",
    "        bhk_stats = {}\n",
    "        for bhk, bhk_df in location_df.groupby('bhk'):\n",
    "            bhk_stats[bhk] = {\n",
    "                'mean': np.mean(bhk_df.price_per_sqft),\n",
    "                'std': np.std(bhk_df.price_per_sqft),\n",
    "                'count': bhk_df.shape[0]\n",
    "            }\n",
    "        for bhk, bhk_df in location_df.groupby('bhk'):\n",
    "            stats = bhk_stats.get(bhk-1)\n",
    "            if stats and stats['count']>5:\n",
    "                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)\n",
    "    return df.drop(exclude_indices,axis='index')\n",
    "df8 = remove_bhk_outliers(df7)\n",
    "# df8 = df7.copy()\n",
    "df8.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "c9905cb9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Count')"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib\n",
    "matplotlib.rcParams[\"figure.figsize\"] = (20,10)\n",
    "plt.hist(df8.price_per_sqft,rwidth=0.8)\n",
    "plt.xlabel(\"Price Per Square Feet\")\n",
    "plt.ylabel(\"Count\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "89bf342b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4.,  3.,  2.,  5.,  8.,  1.,  6.,  7.,  9., 12., 16., 13.])"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df8.bath.unique()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "6df7f619",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Count')"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(df8.bath,rwidth=0.8)\n",
    "plt.xlabel(\"Number of bathrooms\")\n",
    "plt.ylabel(\"Count\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "a7078e94",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5277</th>\n",
       "      <td>Neeladri Nagar</td>\n",
       "      <td>10 BHK</td>\n",
       "      <td>4000.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>10</td>\n",
       "      <td>4000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8483</th>\n",
       "      <td>other</td>\n",
       "      <td>10 BHK</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>525.0</td>\n",
       "      <td>10</td>\n",
       "      <td>4375.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8572</th>\n",
       "      <td>other</td>\n",
       "      <td>16 BHK</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>550.0</td>\n",
       "      <td>16</td>\n",
       "      <td>5500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9306</th>\n",
       "      <td>other</td>\n",
       "      <td>11 BHK</td>\n",
       "      <td>6000.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>150.0</td>\n",
       "      <td>11</td>\n",
       "      <td>2500.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9637</th>\n",
       "      <td>other</td>\n",
       "      <td>13 BHK</td>\n",
       "      <td>5425.0</td>\n",
       "      <td>13.0</td>\n",
       "      <td>275.0</td>\n",
       "      <td>13</td>\n",
       "      <td>5069.124424</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            location    size  total_sqft  bath  price  bhk  price_per_sqft\n",
       "5277  Neeladri Nagar  10 BHK      4000.0  12.0  160.0   10     4000.000000\n",
       "8483           other  10 BHK     12000.0  12.0  525.0   10     4375.000000\n",
       "8572           other  16 BHK     10000.0  16.0  550.0   16     5500.000000\n",
       "9306           other  11 BHK      6000.0  12.0  150.0   11     2500.000000\n",
       "9637           other  13 BHK      5425.0  13.0  275.0   13     5069.124424"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df8[df8.bath>10]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "9407036a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'count')"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(df8.bath,rwidth=0.8)\n",
    "plt.xlabel(\"number of bathrooms\")\n",
    "plt.ylabel(\"count\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "b06be28b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df9 = df8[df8.bath<df8.bhk+2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "ae3edb90",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>size</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>price_per_sqft</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>4 BHK</td>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "      <td>15017.543860</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>3 BHK</td>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "      <td>11901.840491</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              location   size  total_sqft  bath  price  bhk  price_per_sqft\n",
       "0  1st Block Jayanagar  4 BHK      2850.0   4.0  428.0    4    15017.543860\n",
       "1  1st Block Jayanagar  3 BHK      1630.0   3.0  194.0    3    11901.840491"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df9.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "7c529f13",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(7239, 7)"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df9.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "bc8dc8c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              location  total_sqft  bath  price  bhk\n",
       "0  1st Block Jayanagar      2850.0   4.0  428.0    4\n",
       "1  1st Block Jayanagar      1630.0   3.0  194.0    3\n",
       "2  1st Block Jayanagar      1875.0   2.0  235.0    3"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df10 = df9.drop(['size','price_per_sqft'],axis='columns')\n",
    "df10.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "3c0f7e52",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>6th Phase JP Nagar</th>\n",
       "      <th>7th Phase JP Nagar</th>\n",
       "      <th>8th Phase JP Nagar</th>\n",
       "      <th>9th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "      <th>other</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3 rows × 241 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   1st Block Jayanagar  1st Phase JP Nagar  2nd Phase Judicial Layout  \\\n",
       "0                    1                   0                          0   \n",
       "1                    1                   0                          0   \n",
       "2                    1                   0                          0   \n",
       "\n",
       "   2nd Stage Nagarbhavi  5th Block Hbr Layout  5th Phase JP Nagar  \\\n",
       "0                     0                     0                   0   \n",
       "1                     0                     0                   0   \n",
       "2                     0                     0                   0   \n",
       "\n",
       "   6th Phase JP Nagar  7th Phase JP Nagar  8th Phase JP Nagar  \\\n",
       "0                   0                   0                   0   \n",
       "1                   0                   0                   0   \n",
       "2                   0                   0                   0   \n",
       "\n",
       "   9th Phase JP Nagar  ...  Vishveshwarya Layout  Vishwapriya Layout  \\\n",
       "0                   0  ...                     0                   0   \n",
       "1                   0  ...                     0                   0   \n",
       "2                   0  ...                     0                   0   \n",
       "\n",
       "   Vittasandra  Whitefield  Yelachenahalli  Yelahanka  Yelahanka New Town  \\\n",
       "0            0           0               0          0                   0   \n",
       "1            0           0               0          0                   0   \n",
       "2            0           0               0          0                   0   \n",
       "\n",
       "   Yelenahalli  Yeshwanthpur  other  \n",
       "0            0             0      0  \n",
       "1            0             0      0  \n",
       "2            0             0      0  \n",
       "\n",
       "[3 rows x 241 columns]"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "dummies = pd.get_dummies(df10.location)\n",
    "dummies.head(3)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "94eaccc4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>location</th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1200.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>130.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1st Block Jayanagar</td>\n",
       "      <td>1235.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 245 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "              location  total_sqft  bath  price  bhk  1st Block Jayanagar  \\\n",
       "0  1st Block Jayanagar      2850.0   4.0  428.0    4                    1   \n",
       "1  1st Block Jayanagar      1630.0   3.0  194.0    3                    1   \n",
       "2  1st Block Jayanagar      1875.0   2.0  235.0    3                    1   \n",
       "3  1st Block Jayanagar      1200.0   2.0  130.0    3                    1   \n",
       "4  1st Block Jayanagar      1235.0   2.0  148.0    2                    1   \n",
       "\n",
       "   1st Phase JP Nagar  2nd Phase Judicial Layout  2nd Stage Nagarbhavi  \\\n",
       "0                   0                          0                     0   \n",
       "1                   0                          0                     0   \n",
       "2                   0                          0                     0   \n",
       "3                   0                          0                     0   \n",
       "4                   0                          0                     0   \n",
       "\n",
       "   5th Block Hbr Layout  ...  Vijayanagar  Vishveshwarya Layout  \\\n",
       "0                     0  ...            0                     0   \n",
       "1                     0  ...            0                     0   \n",
       "2                     0  ...            0                     0   \n",
       "3                     0  ...            0                     0   \n",
       "4                     0  ...            0                     0   \n",
       "\n",
       "   Vishwapriya Layout  Vittasandra  Whitefield  Yelachenahalli  Yelahanka  \\\n",
       "0                   0            0           0               0          0   \n",
       "1                   0            0           0               0          0   \n",
       "2                   0            0           0               0          0   \n",
       "3                   0            0           0               0          0   \n",
       "4                   0            0           0               0          0   \n",
       "\n",
       "   Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0                   0            0             0  \n",
       "1                   0            0             0  \n",
       "2                   0            0             0  \n",
       "3                   0            0             0  \n",
       "4                   0            0             0  \n",
       "\n",
       "[5 rows x 245 columns]"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')\n",
    "df11.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "bf401659",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>price</th>\n",
       "      <th>bhk</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>194.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2 rows × 244 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   total_sqft  bath  price  bhk  1st Block Jayanagar  1st Phase JP Nagar  \\\n",
       "0      2850.0   4.0  428.0    4                    1                   0   \n",
       "1      1630.0   3.0  194.0    3                    1                   0   \n",
       "\n",
       "   2nd Phase Judicial Layout  2nd Stage Nagarbhavi  5th Block Hbr Layout  \\\n",
       "0                          0                     0                     0   \n",
       "1                          0                     0                     0   \n",
       "\n",
       "   5th Phase JP Nagar  ...  Vijayanagar  Vishveshwarya Layout  \\\n",
       "0                   0  ...            0                     0   \n",
       "1                   0  ...            0                     0   \n",
       "\n",
       "   Vishwapriya Layout  Vittasandra  Whitefield  Yelachenahalli  Yelahanka  \\\n",
       "0                   0            0           0               0          0   \n",
       "1                   0            0           0               0          0   \n",
       "\n",
       "   Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0                   0            0             0  \n",
       "1                   0            0             0  \n",
       "\n",
       "[2 rows x 244 columns]"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "df12 = df11.drop('location',axis='columns')\n",
    "df12.head(2)\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "77e6e72e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>total_sqft</th>\n",
       "      <th>bath</th>\n",
       "      <th>bhk</th>\n",
       "      <th>1st Block Jayanagar</th>\n",
       "      <th>1st Phase JP Nagar</th>\n",
       "      <th>2nd Phase Judicial Layout</th>\n",
       "      <th>2nd Stage Nagarbhavi</th>\n",
       "      <th>5th Block Hbr Layout</th>\n",
       "      <th>5th Phase JP Nagar</th>\n",
       "      <th>6th Phase JP Nagar</th>\n",
       "      <th>...</th>\n",
       "      <th>Vijayanagar</th>\n",
       "      <th>Vishveshwarya Layout</th>\n",
       "      <th>Vishwapriya Layout</th>\n",
       "      <th>Vittasandra</th>\n",
       "      <th>Whitefield</th>\n",
       "      <th>Yelachenahalli</th>\n",
       "      <th>Yelahanka</th>\n",
       "      <th>Yelahanka New Town</th>\n",
       "      <th>Yelenahalli</th>\n",
       "      <th>Yeshwanthpur</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2850.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1630.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1875.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3 rows × 243 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   total_sqft  bath  bhk  1st Block Jayanagar  1st Phase JP Nagar  \\\n",
       "0      2850.0   4.0    4                    1                   0   \n",
       "1      1630.0   3.0    3                    1                   0   \n",
       "2      1875.0   2.0    3                    1                   0   \n",
       "\n",
       "   2nd Phase Judicial Layout  2nd Stage Nagarbhavi  5th Block Hbr Layout  \\\n",
       "0                          0                     0                     0   \n",
       "1                          0                     0                     0   \n",
       "2                          0                     0                     0   \n",
       "\n",
       "   5th Phase JP Nagar  6th Phase JP Nagar  ...  Vijayanagar  \\\n",
       "0                   0                   0  ...            0   \n",
       "1                   0                   0  ...            0   \n",
       "2                   0                   0  ...            0   \n",
       "\n",
       "   Vishveshwarya Layout  Vishwapriya Layout  Vittasandra  Whitefield  \\\n",
       "0                     0                   0            0           0   \n",
       "1                     0                   0            0           0   \n",
       "2                     0                   0            0           0   \n",
       "\n",
       "   Yelachenahalli  Yelahanka  Yelahanka New Town  Yelenahalli  Yeshwanthpur  \n",
       "0               0          0                   0            0             0  \n",
       "1               0          0                   0            0             0  \n",
       "2               0          0                   0            0             0  \n",
       "\n",
       "[3 rows x 243 columns]"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df12.drop(['price'],axis='columns')\n",
    "X.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "c7c2111a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    428.0\n",
       "1    194.0\n",
       "2    235.0\n",
       "3    130.0\n",
       "4    148.0\n",
       "Name: price, dtype: float64"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y =df12.price\n",
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "10950611",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "da08a896",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8629132245229443"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "lr_clf = LinearRegression()\n",
    "lr_clf.fit(X_train,y_train)\n",
    "lr_clf.score(X_test,y_test)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "cc392690",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.82702546, 0.86027005, 0.85322178, 0.8436466 , 0.85481502])"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "from sklearn.model_selection import ShuffleSplit\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)\n",
    "\n",
    "cross_val_score(LinearRegression(), X, y, cv=cv)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "aaf39ca9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:141: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n",
      "If you wish to scale the data, use Pipeline with a StandardScaler in a preprocessing stage. To reproduce the previous behavior:\n",
      "\n",
      "from sklearn.pipeline import make_pipeline\n",
      "\n",
      "model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())\n",
      "\n",
      "If you wish to pass a sample_weight parameter, you need to pass it as a fit parameter to each step of the pipeline as follows:\n",
      "\n",
      "kwargs = {s[0] + '__sample_weight': sample_weight for s in model.steps}\n",
      "model.fit(X, y, **kwargs)\n",
      "\n",
      "\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_base.py:148: FutureWarning: 'normalize' was deprecated in version 1.0 and will be removed in 1.2. Please leave the normalize parameter to its default value to silence this warning. The default behavior of this estimator is to not do any normalization. If normalization is needed please use sklearn.preprocessing.StandardScaler instead.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n",
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py:359: FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>model</th>\n",
       "      <th>best_score</th>\n",
       "      <th>best_params</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>linear_regression</td>\n",
       "      <td>0.847796</td>\n",
       "      <td>{'normalize': False}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>lasso</td>\n",
       "      <td>0.726740</td>\n",
       "      <td>{'alpha': 2, 'selection': 'random'}</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>decision_tree</td>\n",
       "      <td>0.719500</td>\n",
       "      <td>{'criterion': 'mse', 'splitter': 'best'}</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               model  best_score                               best_params\n",
       "0  linear_regression    0.847796                      {'normalize': False}\n",
       "1              lasso    0.726740       {'alpha': 2, 'selection': 'random'}\n",
       "2      decision_tree    0.719500  {'criterion': 'mse', 'splitter': 'best'}"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "\n",
    "def find_best_model_using_gridsearchcv(X,y):\n",
    "    algos = {\n",
    "        'linear_regression' : {\n",
    "            'model': LinearRegression(),\n",
    "            'params': {\n",
    "                'normalize': [True, False]\n",
    "            }\n",
    "        },\n",
    "        'lasso': {\n",
    "            'model': Lasso(),\n",
    "            'params': {\n",
    "                'alpha': [1,2],\n",
    "                'selection': ['random', 'cyclic']\n",
    "            }\n",
    "        },\n",
    "        'decision_tree': {\n",
    "            'model': DecisionTreeRegressor(),\n",
    "            'params': {\n",
    "                'criterion' : ['mse','friedman_mse'],\n",
    "                'splitter': ['best','random']\n",
    "            }\n",
    "        }\n",
    "    }\n",
    "    scores = []\n",
    "    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)\n",
    "    for algo_name, config in algos.items():\n",
    "        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)\n",
    "        gs.fit(X,y)\n",
    "        scores.append({\n",
    "            'model': algo_name,\n",
    "            'best_score': gs.best_score_,\n",
    "            'best_params': gs.best_params_\n",
    "        })\n",
    "\n",
    "    return pd.DataFrame(scores,columns=['model','best_score','best_params'])\n",
    "\n",
    "find_best_model_using_gridsearchcv(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "670965b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['total_sqft', 'bath', 'bhk', '1st Block Jayanagar',\n",
       "       '1st Phase JP Nagar', '2nd Phase Judicial Layout',\n",
       "       '2nd Stage Nagarbhavi', '5th Block Hbr Layout', '5th Phase JP Nagar',\n",
       "       '6th Phase JP Nagar',\n",
       "       ...\n",
       "       'Vijayanagar', 'Vishveshwarya Layout', 'Vishwapriya Layout',\n",
       "       'Vittasandra', 'Whitefield', 'Yelachenahalli', 'Yelahanka',\n",
       "       'Yelahanka New Town', 'Yelenahalli', 'Yeshwanthpur'],\n",
       "      dtype='object', length=243)"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "3c314ae8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_price(location,sqft,bath,bhk):    \n",
    "    loc_index = np.where(X.columns==location)[0][0]\n",
    "\n",
    "    x = np.zeros(len(X.columns))\n",
    "    x[0] = sqft\n",
    "    x[1] = bath\n",
    "    x[2] = bhk\n",
    "    if loc_index >= 0:\n",
    "        x[loc_index] = 1\n",
    "\n",
    "    return lr_clf.predict([x])[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "a7eb9c1b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "83.86570258312122"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "predict_price('1st Phase JP Nagar',1000, 2, 2)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "93777215",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "86.08062284986892"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "predict_price('1st Phase JP Nagar',1000, 3, 3)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "7ffbcc66",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "193.31197733179854"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "predict_price('Indira Nagar',1000, 2, 2)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "9ba439ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\brnpr\\anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "195.52689759854624"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "predict_price('Indira Nagar',1000, 3, 3)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e13830f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
