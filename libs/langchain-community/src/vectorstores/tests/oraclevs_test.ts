import oracledb from 'oracledb';
import {promises as fs} from "fs";
import {MaxMarginalRelevanceSearchOptions} from "@langchain/core/vectorstores";
import {Document, DocumentInterface} from "@langchain/core/documents";
import {Embeddings} from "@langchain/core/embeddings";
import {Callbacks} from "@langchain/core/callbacks/manager";
import {DistanceStrategy, createIndex, OracleVS} from "../oraclevs.js";
import { HuggingFaceTransformersEmbeddings } from "../../embeddings/hf_transformers.js";

interface DataRow {
  id: string;
  link: string;
  text: string;
}

async function dbConnect(): Promise<oracledb.Connection> {
  // Create a connection
  const connection = await oracledb.getConnection({
    user: 'shailendra',
    password: 'shailendra1',
    connectString: '138.2.233.65:1521/freepdb1'
  });
  console.log('Connection pool started')
  return connection

}

async function dbPool(): Promise<oracledb.Pool> {
  // Create a connection pool
  const pool = await oracledb.createPool({
    user: 'shailendra',
    password: 'shailendra1',
    connectString: '138.2.233.65:1521/freepdb1'
  });

  console.log('Connection pool started')
  return pool
}


class TestsOracleVS {
  client: any | null = null;

  docsDir = "/Users/skmishra/repo/frameworks/oraclevsjs/resources/downloads/oradocs/";

  filename: string;

  embeddingFunction: HuggingFaceTransformersEmbeddings;

  dbConfig: Record<string, any>= {};

  oraclevs!: OracleVS;

  constructor(filename: string,
              embeddingFunction: HuggingFaceTransformersEmbeddings) {
    this.filename = filename;
    this.embeddingFunction = embeddingFunction;
  }

  async init(): Promise<void> {
    this.client = await dbPool()

    // code to create dbConfig
    this.dbConfig = {
      "client": this.client,
      "tableName": "some_tablenm",
      "distanceStrategy": DistanceStrategy.DOT_PRODUCT,
      "query": "What are salient features of oracledb",
    };

    try {
      this.oraclevs = new OracleVS(this.embeddingFunction, this.dbConfig);
    } catch (error) {
      console.error("An exception occurred ::", error);
      // Handle error
    }
  }

  private createDocument(row: DataRow): Document {
    const metadata = {
      id: row.id,
      link: row.link,
    };
    return new Document({pageContent: row.text, metadata});
  }

  public async testIngestJson(): Promise<Document[]> {
    try {
      const filePath = `${this.docsDir}${this.filename}`;
      const fileContent = await fs.readFile(filePath, {encoding: 'utf8'});
      const jsonData: DataRow[] = JSON.parse(fileContent);
      return jsonData.map((row) => this.createDocument(row));
    } catch (error) {
      console.error('An error occurred while ingesting JSON:', error);
      throw error; // Rethrow the error if you want the calling function to handle it
    }
  }

  public async testCreateIndex(): Promise<void> {
    try {
      const connection : oracledb.Connection= await dbConnect()

      await createIndex(connection, this.oraclevs, {
        idxName: "IVF",
        idxType: "IVF",
        neighborPart: 64,
        accuracy: 90
      });

      console.log("Index created successfully");
      await connection.close();
    } catch (ex) {
      console.error("Exception occurred while index creation", ex);
      // TypeScript/JavaScript does not have a direct equivalent to Python's traceback.print_exc(),
      // so we log the error object directly, which includes the stack trace.
    }
  }

  public async testSimilaritySearchByVector(
    embedding: number[],
    k: number,
    filter?: OracleVS["FilterType"],
  ): Promise<[DocumentInterface, number][]> {
    return this.oraclevs.similaritySearchVectorWithScore(
      embedding,
      k,
      filter,
    );
  }

  public async testSimilaritySearchByVectorReturningEmbeddings(
    embedding: number[],
    // eslint-disable-next-line default-param-last
    k = 4,
    filter?: OracleVS["FilterType"],
  ): Promise<[Document, number, Float32Array | number[]][]> {
    return await this.oraclevs.similaritySearchByVectorReturningEmbeddings( embedding, k, filter);
  }

  public async testMaxMarginalRelevanceSearch(
    query: string,
    options?: MaxMarginalRelevanceSearchOptions<OracleVS["FilterType"]>,
    _callbacks?: Callbacks
  ): Promise<DocumentInterface[]> {
    if (!options) {
      // eslint-disable-next-line no-param-reassign
      options = { k: 10, fetchK: 20 }; // Default values for the options
    }
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    return this.oraclevs.maxMarginalRelevanceSearch(query, options, _callbacks);
  }

  public async testMaxMarginalRelevanceSearchByVector(
    query: number[],
    options?: MaxMarginalRelevanceSearchOptions<OracleVS["FilterType"]>,
    _callbacks?: Callbacks | undefined
  ): Promise<DocumentInterface[]> {
    if (!options) {
      // eslint-disable-next-line no-param-reassign
      options = { k: 10, fetchK: 20 }; // Default values for the options
    }
    return this.oraclevs!.maxMarginalRelevanceSearchByVector(query, options, _callbacks);
  }

  public async testMaxMarginalRelevanceSearchWithScoreByVector(
    embedding: number[],
    options?: MaxMarginalRelevanceSearchOptions<OracleVS["FilterType"]>,
    _callbacks?: Callbacks | undefined
  ): Promise<Array<{ document: Document; score: number }>> {
    if (!options) {
      // eslint-disable-next-line no-param-reassign
      options = { k: 10, fetchK: 20 }; // Default values for the options
    }
    return this.oraclevs.maxMarginalRelevanceSearchWithScoreByVector(embedding, options, _callbacks)
  }

  testDelete( params: { ids?: string[], deleteAll?: boolean } ): Promise<void> {
    return this.oraclevs.delete(params);
  }
}

async function runTestsOracleVS() {
  // Initialize dotenv to load environment variables
  const query = "What is the language used by Oracle database";

  // Set up the embedding function model: "Xenova/all-MiniLM-L6-v2"
  const embeddingFunction = new HuggingFaceTransformersEmbeddings();
  if (!embeddingFunction) {
    console.error("Failed to initialize the embedding function.");
    return;
  }

  // eslint-disable-next-line no-instanceof/no-instanceof
  if (!(embeddingFunction instanceof Embeddings)) {
    console.error("Embedding function is not an instance of Embeddings.");
    return;
  }

  console.log("Embedding function initialized successfully");

  // Initialize the TestsOracleVS class
  const testsOracleVS = new TestsOracleVS("concepts23c_small.json",
    embeddingFunction);

  // Initialize connection and other setup
  await testsOracleVS.init();

  // Ingest JSON data to create documents
  const documents = await testsOracleVS.testIngestJson();
  await OracleVS.fromDocuments(documents,
    testsOracleVS.embeddingFunction,
    testsOracleVS.dbConfig)

  // Create an index
  await testsOracleVS.testCreateIndex();

  // Assume some dummy embedding vector for demonstration
  // const embedding: number[] = [0.1, 0.2, 0.3, 0.4]; // Example embedding

  // Perform a similarity search by vector
  const embedding = await embeddingFunction.embedQuery(query);
  const similaritySearchByVector = await testsOracleVS.testSimilaritySearchByVector(embedding, 5);
  console.log("Similarity Search Results:", similaritySearchByVector);

  // Perform a similarity search by vector
  const similaritySearchByEmbeddings =
    await testsOracleVS.testSimilaritySearchByVectorReturningEmbeddings(embedding, 5)
  console.log("Similarity Search Results:", similaritySearchByEmbeddings);

  const maxMarginalRelevanceSearch =
    await testsOracleVS.testMaxMarginalRelevanceSearch(query)
  console.log("Max Marginal Relevance Search:", maxMarginalRelevanceSearch);

  const maxMarginalRelevanceSearchByVector =
    await testsOracleVS.testMaxMarginalRelevanceSearchByVector(embedding)
  console.log("Max Marginal Relevance Search By Vector:", maxMarginalRelevanceSearchByVector);

  const maxMarginalRelevanceSearchWithScoreByVector =
    await testsOracleVS.testMaxMarginalRelevanceSearchWithScoreByVector(embedding)
  console.log("Max Marginal Relevance Search By Vector:", maxMarginalRelevanceSearchWithScoreByVector);

}

// Run the demonstration
runTestsOracleVS().catch(console.error);