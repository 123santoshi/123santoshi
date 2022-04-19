{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "sales prediction",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOPGtnA4hXRASpmNE67t/r0",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/123santoshi/123santoshi/blob/main/sales_prediction.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tP2s8Mt1jgiU"
      },
      "source": [
        "!pip install pandas      #pandas installation\n",
        "!pip install numpy        #numpy installation\n",
        "!pip install scikit-learn #sklearn installation"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YDfHXqV0lAhH"
      },
      "source": [
        "# LOAD DATASET\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JS437dmij273"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "data={\n",
        "    \"age\":[20,30,40,56,68,19,39,49,69,57,10,23,25,67,89,12,34,57,78,98,20,30,40,56,68,19,39,49,69,57,10,23,25,67,89,12,34,57,78,98,20,30,40,56,68,19,39,49,69,57,10,23,25,67,89,12,34,57,78,98,20,30,40,56,68,19,39,49,69,57,10,23,25,67,89,12,34,57,78,98],\n",
        "    \"salary\":[10000,20000,3000000,56000,450000,320000,25000,68000,58000,35000,158000,25000,3500000,56000,40000,30000,45000,86000,65000,45000,10000,20000,3000000,56000,450000,320000,25000,68000,58000,35000,158000,25000,3500000,56000,40000,30000,45000,86000,65000,45000,10000,20000,3000000,56000,450000,320000,25000,68000,58000,35000,158000,25000,3500000,56000,40000,30000,45000,86000,65000,45000,10000,20000,3000000,56000,450000,320000,25000,68000,58000,35000,158000,25000,3500000,56000,40000,30000,45000,86000,65000,45000],\n",
        "    \"buy\":[0,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0]\n",
        "}\n",
        "df=pd.DataFrame(data)\n",
        "df"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**SUMMARIZE THE DATASET**\n"
      ],
      "metadata": {
        "id": "_yFsqZqHV6Jf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(df.shape)  #no of rowa and no of columns"
      ],
      "metadata": {
        "id": "MtCAPTRuTf7F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "id": "GnWbf-AZWG5Z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.tail()"
      ],
      "metadata": {
        "id": "RxzGX25zWMd2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ts-3sGU5lHiP"
      },
      "source": [
        "x=df.iloc[:,:-1]       #x takes input variables based on age,salary we predict whether the person buys or not so age,salary are input values\n",
        "print(x)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wtOF1jz9lm3q"
      },
      "source": [
        "y=df.iloc[:,-1]            #Y takes output value\n",
        "print(y)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "T-64jlnGlqG6"
      },
      "source": [
        "from sklearn.model_selection import train_test_split         \n",
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)      #training data=75% test data=25%"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WthBVFE1mhYV"
      },
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "sc= StandardScaler() \n",
        "x_train=sc.fit_transform(x_train)         #we have to transofrm our data into standarad format\n",
        "x_test=sc.transform(x_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EJ6t6BD1nYnG"
      },
      "source": [
        "from sklearn.linear_model import LogisticRegression  #we are using classification so import Logistic Regression\n",
        "reg=LogisticRegression()\n",
        "reg.fit(x_train,y_train)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fJ0AeXmXnrly"
      },
      "source": [
        "age=int(input(\"enter age\"))\n",
        "salary=int(input(\"enter salary\"))\n",
        "newcustomer=[[age,salary]]\n",
        "result=reg.predict(sc.transform(newcustomer))\n",
        "print(result)\n",
        "if(result==0):\n",
        "  print(\"person cannot buy the product\")\n",
        "else:\n",
        "  print(\"person can buy the product\")\n"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}