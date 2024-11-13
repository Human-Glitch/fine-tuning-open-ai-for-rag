import pandas as pd
import json

class Preprocessor:
    def json_to_dataframe_with_titles(json_data):
        qas = []
        context = []
        is_impossible = []
        answers = []
        titles = []

        for article in json_data['data']:
            title = article['title']
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    qas.append(qa['question'].strip())
                    context.append(paragraph['context'])
                    is_impossible.append(qa['is_impossible'])
                    
                    ans_list = []
                    for ans in qa['answers']:
                        ans_list.append(ans['text'])
                    answers.append(ans_list)
                    titles.append(title)

        df = pd.DataFrame({'title': titles, 'question': qas, 'context': context, 'is_impossible': is_impossible, 'answers': answers})
        return df

    def get_diverse_sample(df, sample_size=100, random_state=42):
        """
        Get a diverse sample of the dataframe by sampling from each title
        """
        sample_df = df.groupby(['title', 'is_impossible']).apply(lambda x: x.sample(min(len(x), max(1, sample_size // 50)), random_state=random_state)).reset_index(drop=True)
        
        if len(sample_df) < sample_size:
            remaining_sample_size = sample_size - len(sample_df)
            remaining_df = df.drop(sample_df.index).sample(remaining_sample_size, random_state=random_state)
            sample_df = pd.concat([sample_df, remaining_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        return sample_df.sample(min(sample_size, len(sample_df)), random_state=random_state).reset_index(drop=True)

    def dataframe_to_jsonl(df):
        def create_jsonl_entry(row):
            answer = row["answers"][0] if row["answers"] else "I don't know"
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"""Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
                Question: {row.question}\n\n
                Context: {row.context}\n\n
                Answer:\n""",
                },
                {"role": "assistant", "content": answer},
            ]
            return json.dumps({"messages": messages})

        jsonl_output = df.apply(create_jsonl_entry, axis=1)
        return "\n".join(jsonl_output)