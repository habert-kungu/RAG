import { MongoClient } from "mongodb";
import {
  RecursiveCharacterTextSplitter,
  TextSplitter,
} from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "langchain/community/vectorstores/mongodb_atlas";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { createMetadataTagger } from "@langchain/community/document_transformers/openai_functions";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";

// setting up the mongodb connections

async function fileLoader() {
  // setting up mongodb connections
  const client = new MongoClient(process.env.MONGODB_URI);
  await client.connect();
  const collection = client
    .db("book_mongodb_chunks")
    .collection("chunked_data");

  const loader = PDFLoader(".\sample_files\mongodb.pdf");
  const pages = loader.load();
  const cleanedPages = [];

  for (let page in pages) {
    if (page.page_content.split(" ").length > 20) {
      cleanedPages.push(page);
    }
  }
  const textSplitter = RecursiveCharacterTextSplitter(
    (chunk_size = 500),
    (chunk_overlap = 150),
  );

  const schema = {
    properties: {
      title: { type: "string" },
      keywords: { type: "array", items: { type: "string" } },
      hasCode: { type: "boolean" },
    },
    required: ["title", "keywords", "hasCode"],
  };
  const llm = new ChatGoogleGenerativeAI(
    (geminiApiKey = process.env.GEMINI_API_KEY),
    (temperature = 0),
    (model = "gemini-2.0-flash "),
  );

  const document_transformer = createMetadataTagger(
    (metaDataSchema = schema),
    (llm = llm),
  );
  const docs = document_transformer.transform_documents(cleanedPages);
  const splitDocs = textSplitter.split(docs);
  const embeddings = GoogleGenerativeAIEmbeddings(geminiApiKey);
  const vectorStore = MongoDBAtlasVectorSearch.from(
    splitDocs,
    embeddings,
    (collection = collection),
  );
}
