import argparse
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset", type=str, required=True)
    parser.add_argument("--train_result", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    st.write(f"Args: {args}")
    investigate_dataset(args.original_dataset)
    describe_train_process(args.train_result, args.model)


def describe_train_process(train_csv: str, model_path: str):
    df = pd.read_csv(train_csv)
    st.header("Training process")
    st.subheader("Results of training")
    st.write(df)

    column_keys = set()
    prefixes = ["train_", "val_"]
    for col in df.columns:
        if any(col.startswith(p) for p in prefixes):
            _, _, key = col.partition("_")
            column_keys.add(key)
    for col in column_keys:
        st.subheader(col)
        st.plotly_chart(px.line(df, y=[p + col for p in prefixes], x="epoch"))
    st.markdown(f"**Model path:** {model_path}")


def investigate_dataset(original_dataset: str):
    df = pd.read_csv(original_dataset)
    st.header("Original dataset insights")
    st.subheader("Samples of dataset")
    st.write(df.head(20))

    st.subheader("Distributions of labels")
    st.plotly_chart(px.pie(df, "label"))

    st.subheader("Distributions of texts' lengths")
    df["text_len"] = df.text.apply(lambda x: len(x))
    st.plotly_chart(
        px.histogram(df, x="text_len", color="label", opacity=0.75, barmode="overlay")
    )


if __name__ == "__main__":
    main()
