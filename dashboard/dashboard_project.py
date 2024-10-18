# Import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from babel.numbers import format_currency
from streamlit_folium import st_folium
sns.set_style('dark')

#Create all functions
def format_with_units(x, pos): # untuk mengubah format plot menjadi dalam bentuk ribuan (K) atau jutaaan (M)
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.1f}K'
    else:
        return f'{x:.0f}'

def create_daily_orders_df(df): # total order dan revenues dalam bentuk plot yang sesuai input date range
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        'order_id': 'nunique',
        'payment_value': 'sum'
    }, inplace=True)
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        'order_id': 'order_count',
        'payment_value': 'revenue'
    }, inplace=True)
    return daily_orders_df

def create_sum_order_items_df(df): # sum of order item by product_category
    sum_orders_items_df = df.groupby('product_category_name_english')['order_item_id'].sum().reset_index()
    sum_orders_items_df = sum_orders_items_df.sort_values(by='order_item_id', ascending=False).reset_index()
    return sum_orders_items_df

def create_sum_revenues_items_df(df): # sum of revenues by product category
    sum_revenues_items_df = df.groupby('product_category_name_english')['payment_value'].sum().reset_index()
    sum_revenues_items_df = sum_revenues_items_df.sort_values(by='payment_value', ascending=False).reset_index()
    return sum_revenues_items_df

def create_sum_order_items_state_df(df):# sum of order item by state
    sum_orders_items_state_df = df.groupby(['customer_state']).agg({
        'order_item_id': 'sum'
    }).sort_values(by='order_item_id', ascending=False).reset_index()
    return sum_orders_items_state_df

def create_sum_revenues_items_state_df(df): # sum of revenues by state
    sum_revenues_items_state_df = df.groupby(['customer_state']).agg({
        'payment_value': 'sum'
    }).sort_values(by='payment_value', ascending=False).reset_index()
    return sum_revenues_items_state_df

def create_bystate_df(df): # sum of customer by state
    bystate_df = df.groupby("customer_state")['customer_id'].nunique().reset_index()
    bystate_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    return bystate_df

def create_bypaymenttype_df(df): # sum of using payment_type category by payment value
    payment_type_df = df.groupby(['customer_state', 'payment_type'])['payment_value'].sum().unstack()
    payment_type_df['total'] = payment_type_df.sum(axis=1)
    payment_type_df = payment_type_df.sort_values(by='total', ascending=False).drop(columns='total')
    return payment_type_df

def create_transactioncounts_df(df): # sum of using payment type category by state
    transaction_counts_df = df.groupby(['customer_state', 'payment_type']).size().unstack()
    transaction_counts_df['total'] = transaction_counts_df.sum(axis=1)
    transaction_counts_df = transaction_counts_df.sort_values(by='total', ascending=False).drop(columns='total')
    return transaction_counts_df

def create_rfm_df(df): # create rfm analysis
    rfm_df = df.groupby('customer_id').agg({
        'payment_value' : 'sum',
        'order_id' : 'nunique',
        'order_purchase_timestamp' : 'max'
    }).reset_index()
    rfm_df.rename(columns={
        'payment_value' : 'monetary',
        'order_id' : 'frequency',
        'order_purchase_timestamp' : 'max_order_timestamp'
    }, inplace=True)
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df['order_purchase_timestamp'].dt.date.max()
    rfm_df['recency'] = rfm_df['max_order_timestamp'].apply(lambda x: (recent_date - x).days)
    rfm_df.drop(columns='max_order_timestamp', axis=1, inplace=True)
    return rfm_df

def create_customer_seg_df(df): # create clustering for customer
    customer_seg_df = df.groupby('customer_id').agg({
        'payment_value' : 'sum',
    }).reset_index()
    customer_seg_df.rename(columns={
        'payment_value' : 'monetary',
    }, inplace=True)
    bins = [0, 50, 200, 500, 1000, 5000, float('inf')]
    customer_value_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Rich Loyalist']
    customer_seg_df['spending_category'] = pd.cut(customer_seg_df['monetary'], bins=bins, labels=customer_value_labels, right=False)
    customer_seg_df = customer_seg_df['spending_category'].value_counts().reset_index()
    customer_seg_df = customer_seg_df.sort_values(by='count', ascending=False)
    return customer_seg_df

def create_customer_geolocation_df(df, start_date, end_date): # create a customer geolocation in a map using folium
    df = df.groupby(
        ['geolocation_lat', 'geolocation_lng', 'geolocation_city', 'geolocation_state']).size().reset_index(
        name='customer_count')
    # If there are no records in the filtered range, return an empty map with the default location
    if df.empty:
        return folium.Map(location=[0, 0], zoom_start=2)
    # Get the mean latitude and longitude for centering the map
    center_lat = df['geolocation_lat'].mean()
    center_lng = df['geolocation_lng'].mean()
    # Create a folium map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)
    # Add circle markers
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['geolocation_lat'], row['geolocation_lng']],
            radius=5,  # Base size of the circle
            popup=f"City: {row['geolocation_city']}, State: {row['geolocation_state']}<br>Customers: {row['customer_count']}",
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            weight=2
        ).add_to(marker_cluster)
    return m

#Read csv yang sudah di-export di jupyter notebook
all_df = pd.read_csv("all_df.csv")

# Mengubah datatype column menjadi datetime64ns
datetime_columns = ['order_purchase_timestamp', 'order_estimated_delivery_date']
all_df.sort_values(by='order_purchase_timestamp', inplace=True)
all_df.reset_index(inplace=True)
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Membuat komponen filter
min_date = all_df['order_purchase_timestamp'].min()
max_date = all_df['order_purchase_timestamp'].max()

st.set_page_config(layout="wide")


#Membuat sidebar
with st.sidebar:
    # Menambah logo perusahaan
    st.image("https://raw.githubusercontent.com/doni2123/E-commerce-Dashboard/refs/heads/main/dashboard/company_logo.png")
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Hasil filter disimpan ke dalam main_df
main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) &
                 (all_df["order_purchase_timestamp"] <= str(end_date))]

# Hasil
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
sum_revenues_items_df = create_sum_revenues_items_df(main_df)
sum_order_items_state_df = create_sum_order_items_state_df(main_df)
sum_revenues_items_state_df = create_sum_revenues_items_state_df(main_df)
bystate_df = create_bystate_df(main_df)
bypaymenttype_df = create_bypaymenttype_df(main_df)
transactioncounts_df = create_transactioncounts_df(main_df)
rfm_df = create_rfm_df(main_df)
customer_seg_df = create_customer_seg_df(main_df)
customer_geolocation_df = create_customer_geolocation_df(main_df, start_date, end_date)


# Visulisasi
st.header('Dicoding E-Commerce Collection Dashboard :sparkles:')

st.subheader('Daily Orders & Revenues')


# Pertanyaan 1: Bagaimana performa penjualan pada beberapa tahun terakhir dalam skala per bulan? (performa banyaknya order dan total revenue per month)
c1, c2= st.columns([1,1])
with c1:
    total_orders = daily_orders_df['order_count'].sum()
    st.metric('Total orders: ', value=total_orders)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(
        daily_orders_df['order_purchase_timestamp'],
        daily_orders_df['order_count'],
        marker='o', markersize=2,
        linewidth=2
        )
    ax.set_title("Total Orders", loc="center", fontsize=14)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.grid(linestyle='--', alpha= 1, which='major', axis='y')
    ax.grid(linestyle='dotted', alpha= 1, which='major', axis='x')
    st.pyplot(fig)

with c2:
    total_revenue = format_currency(daily_orders_df['revenue'].sum(), 'BRL', locale='es_CO')
    st.metric(label='Total revenues: ', value=total_revenue, )
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(
    daily_orders_df['order_purchase_timestamp'],
        daily_orders_df['revenue'],
        marker ='o',
        markersize=2,
        color = 'seagreen',
        linewidth=2
        )
    ax.set_title("Total Revenues", loc="center", fontsize=14)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8, rotation=45, direction='out')
    ax.yaxis.set_major_formatter(FuncFormatter(format_with_units))
    ax.grid(linestyle='--', alpha= 1, which='major', axis='y')
    ax.grid(linestyle='dotted', alpha= 1, which='major', axis='x')
    st.pyplot(fig)


# Pertanyaan 2: Produk apa yang paling menghasilkan revenue dan paling laku serta produk paling tidak laku dan tidak menghasilkan revenue?
st.subheader('Best & Worst Performing Products')

max_order_items = sum_order_items_df['order_item_id'].max()
max_payment = sum_revenues_items_df['payment_value'].max()
max_x_order = max_order_items * 1.1
max_x_payment = max_payment * 1.1

col1, col2 = st.columns(2)
with col1:
    tab1,tab2 = st.tabs(['Revenues', 'Orders'])
    with tab1:
        fig, ax = plt.subplots(2,1, figsize=(8,18))

        sns.barplot(y='product_category_name_english', x='payment_value', data=sum_revenues_items_df.head(15), ax=ax[0], hue='payment_value')
        ax[0].set_ylabel(None)
        ax[0].set_xlabel('Sum of Revenues in millions')
        ax[0].tick_params(axis ='both', labelsize=14)
        ax[0].set_title("15 Best Performing Product Based on Revenues", loc="center", fontsize=20)
        ax[0].set_xlim(0, max_x_payment)

        sns.barplot(y='product_category_name_english', x='payment_value', data=sum_revenues_items_df.tail(20).sort_values(by='payment_value',ascending=False), ax=ax[1], color= '#e9d4d0')
        ax[1].set_ylabel(None)
        ax[1].set_xlabel('Sum of Revenues in millions')
        ax[1].tick_params(axis ='both', labelsize=14)
        ax[1].set_title("20 Worst Performing Product Based on Revenues", loc="center", fontsize=20)
        ax[1].set_xlim(0, max_x_payment)

        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(2,1, figsize=(8,18))
        sns.barplot(y='product_category_name_english', x='order_item_id', data=sum_order_items_df.head(15), ax=ax[0],hue="order_item_id")
        ax[0].set_ylabel(None)
        ax[0].tick_params(axis ='both', labelsize=14)
        ax[0].set_xlabel('Sum of order items')
        ax[0].set_title("15 Best Performing Product Based on Order Items", loc="center", fontsize=20)
        ax[0].set_xlim(0, max_x_order)

        sns.barplot(y='product_category_name_english', x='order_item_id', data=sum_order_items_df.tail(20).sort_values(by='order_item_id',ascending=False), ax=ax[1], color= '#e9d4d0')
        ax[1].set_ylabel(None)
        ax[1].set_xlabel('Sum of order items')
        ax[1].tick_params(axis= 'both', labelsize=14)
        ax[1].set_title("20 Worst Performing Product Based on Order Items", loc="center", fontsize=20)
        ax[1].set_xlim(0, max_x_order)

        st.pyplot(fig)


# Pertanyaan 3: State apa yang menghasilkan revenue dan jumlah order paling tinggi?
with col2:
    st.subheader('Total Order Items & Revenues Based on States')
    tab1,tab2 = st.tabs(['Revenues', 'Orders'])
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 20))
        sns.barplot(y='customer_state', x='order_item_id', data=sum_order_items_state_df, hue="order_item_id")
        ax.set_ylabel(None)
        ax.tick_params(axis ='both', labelsize=14)
        ax.set_xlabel('Sum of order items')
        ax.set_title("Total Order Items Based on States", loc="center", fontsize=20)
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 20))
        sns.barplot(y='customer_state', x='payment_value', data=sum_revenues_items_state_df, hue="payment_value")
        ax.set_ylabel(None)
        ax.set_xlabel('Sum of Revenues in millions')
        ax.tick_params(axis ='both', labelsize=14)
        ax.set_title("Total Revenues Based on States", loc="center", fontsize=20)
        st.pyplot(fig)


col1, col2, col3 = st.columns([2,1,1])
with col1:
    # Pertanyaan 4: Metode pembayaran apa yang paling banyak jumlah pembayarannya dan sering digunakan  dari berbagai states?
    st.subheader('The Most Used Payment Type based on Total Amount Spent and Total Orders')
    tab1,tab2 = st.tabs(['Revenues', 'Orders'])
    with tab1:
        bypaymenttype_df.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title('Total Ammount Used on Payment Each Payment Type by Customer State', loc="center", fontsize=15)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_with_units))
        plt.grid(linestyle='-', alpha=1, which='major', axis='y')
        plt.grid(linestyle='dotted', alpha=1, which='major', axis='x')
        plt.legend(title='Payment Type', loc='upper right')
        st.pyplot(plt.gcf())
    with tab2:
        transactioncounts_df.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title('Number of Orders for Each Payment Type by Customer State',  loc="center", fontsize=15)
        plt.grid(linestyle='-', alpha=1, which='major', axis='y')
        plt.grid(linestyle='dotted', alpha=1, which='major', axis='x')
        plt.legend(loc='upper right', title='Payment Type')
        st.pyplot(plt.gcf())


with col2:
    # Pertanyaan 5: RFM Analysis
    st.subheader('RFM Analysis')
    st.markdown("Best Customer Based on RFM Parameters (customer_id)")

    tab1, tab2, tab3 = st.tabs(["Recency", "Monetary", "Frequency"])
    with tab1:
        fig, ax = plt.subplots(1,1)
        sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5))
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.set_title("By Recency (days)", loc="center", fontsize=12)
        ax.tick_params(axis='x', labelsize=10, rotation=90)
        ax.grid(linestyle='-', alpha=1, which='major', axis='y')
        ax.grid(linestyle='dotted', alpha=1, which='major', axis='x')
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5))
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.set_title("By Frequency", loc="center", fontsize=12)
        ax.tick_params(axis='x', labelsize=10, rotation=90)
        ax.grid(linestyle='-', alpha=1, which='major', axis='y')
        ax.grid(linestyle='dotted', alpha=1, which='major', axis='x')
        st.pyplot(fig)
    with tab3:
        fig, ax = plt.subplots(1, 1)
        sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5))
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.set_title("By Monetary", loc="center", fontsize=12)
        ax.tick_params(axis='x', labelsize=10, rotation=90)
        ax.grid(linestyle='-', alpha=1, which='major', axis='y')
        ax.grid(linestyle='dotted', alpha=1, which='major', axis='x')
        st.pyplot(fig)

with col3:
    # Pertanyaan 6: Kategori customer berdasarkan jumlah total pembayaran
    st.subheader('Customer Categorization Based on Total Spending Value')

    fig, ax = plt.subplots(1, 1)
    sns.barplot(x = 'spending_category', y= 'count', data = customer_seg_df, hue='count', legend=False)
    ax.set_title("Customer Categorization", loc="center", fontsize=12)
    ax.set_xlabel('Customer Category')
    ax.set_ylabel('Number of customer')
    ax.grid(linestyle='--', alpha= 1, which='major', axis='y')
    ax.grid(linestyle='dotted', alpha= 1, which='major', axis='x')
    for i in ax.containers:
        ax.bar_label(i,)
    st.pyplot(fig)
    with st.expander('Criteria'):
        category = {'Customer Category': ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Rich Loyalist'],
                    'Range Value (BRL)': ['0-50', '50-200', '200-500', '500-1000', '1000-5000', '>5000']}
        category_df = pd.DataFrame(category)
        st.dataframe(category_df)


# Pertanyaan 7: Distribusi customer berdasarkan lokasi geografis dalam bentuk map
st.subheader('Customer Geographical Distribution by States')
tab1,tab2 = st.tabs(['By Map', 'By Chart'])
with tab1:
    # Buat map dengan memanggil fungsi yang sudah dibuat
    m = create_customer_geolocation_df(main_df, start_date, end_date)
    # Display the map in Streamlit
    st_folium(m)
with tab2:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.barplot(x='customer_state', y='customer_count', data=bystate_df.sort_values(by='customer_count', ascending=False), hue='customer_count')
    ax.set_title("Number of Customers by States", loc="center", fontsize=20)
    ax.set_xlabel('States')
    ax.set_ylabel(None)
    st.pyplot(fig)

st.caption('Doni Maulana Syahputra 2024')