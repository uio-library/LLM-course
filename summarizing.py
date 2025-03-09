import argparse
import os
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.output_parsers import RegexParser
import torch
import re


def load_model(model_id):
    '''
    Load the model with the given model_id.
    The HuggingFace library will automatically download the model
    if it is not already downloaded.
    If a GPU is available, use it, otherwise use the CPU.
    '''
    # Check if a GPU is available
    device = 0 if torch.cuda.is_available() else -1
    model = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task='text-generation',
        device=device,
        pipeline_kwargs={
            'max_new_tokens': 1000,
            # 'do_sample': True,
            # 'temperature': 0.3,
            # 'num_beams': 4,
        }
    )
    return model


def parse_arguments():
    '''Parse command-line arguments'''
    parser = argparse.ArgumentParser(
        description='Summarize text files in the folder')
    parser.add_argument('input', help='Folder containing text files')
    parser.add_argument('--output', default='summaries.txt',
                        help='Where to save the summaries. If a directory is given, each summary will be saved to a separate file in the directory.')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Model to use for summarization')
    parser.add_argument('-v', '--verbose',
                        action='store_true', help='Verbose mode')
    return parser.parse_args()


def load_documents(input_folder):
    '''Load all documents in input_folder'''
    loader = DirectoryLoader(input_folder)
    documents = loader.load()
    print('number of documents:', len(documents))
    return documents


def make_prompt(separator):
    '''
    Create a prompt template to summarize the input text.
    The prompt template has a variable 'context' that will be replaced
    with the input text.
    '''
    prompt_template = '''Write a summary of the following:
    {context}
    ''' + separator
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context'])

    return prompt


def main():
    # Set the cache directory for HuggingFace models
    os.environ['HF_HOME'] = '/fp/projects01/ec443/huggingface/cache/'

    # parse command-line arguments
    args = parse_arguments()

    if args.verbose:
        print('Input folder:', args.input)
        print('Output folder or file:', args.output)
        print('Model:', args.model)

    # load the model
    model = load_model(args.model)

    # the separator is used to separate the input text from the summary
    separator = '\nYour Summary:\n'
    prompt = make_prompt(separator)

    # Define the regex pattern to extract the summary
    output_parser = RegexParser(
        regex=rf'{separator}(.*)', output_keys=['summary'], flags=re.DOTALL)

    chain = create_stuff_documents_chain(
        model, prompt, output_parser=output_parser)

    # Loading the Documents
    documents = load_documents(args.input)

    # Creating the Summaries
    summaries = {}

    for document in documents:
        filename = document.metadata['source']
        print('Summarizing document:', filename)
        result = chain.invoke({"context": [document]})
        summary = result['summary']
        summaries[filename] = summary
        print('Summary of file', filename)
        print(summary)

    # Saving the Summaries to a Text File
    # check if output is a directory
    if os.path.isdir(args.output):
        # save each summary to a separate file in the output directory
        for filename in summaries:
            # get the file name without the directory
            basename = os.path.basename(filename)
            name = os.path.join(args.output, basename + '-summary.txt')
            if args.verbose:
                print('Saving summary to', name)
            with open(name, 'w') as outfile:
                print('Summary of ', filename, file=outfile)
                print(summaries[filename], file=outfile)
    else:
        # save all summaries to a single file
        with open(args.output, 'w') as outfile:
            for filename in summaries:
                print('Summary of ', filename, file=outfile)
                print(summaries[filename], file=outfile)
                print(file=outfile)


if __name__ == '__main__':
    main()
