/* eslint-disable no-process-env */
import { test, expect } from "@jest/globals";
import { Document } from "@langchain/core/documents";
import oracledb from "oracledb";
import { env } from "node:process";
import { createHash } from "crypto";
import {
  OracleEmbeddings,
  createIndex,
  dropTablePurge,
  DistanceStrategy,
  type OracleDBVSStoreArgs,
} from "../index.js";
import { OracleVS, type Metadata } from "../vectorstore.js";

describe("OracleVectorStore", () => {
  const tableName = "testlangchain_1";
  let pool: oracledb.Pool;
  let embedder: OracleEmbeddings;
  let connection: oracledb.Connection | undefined;
  let oraclevs: OracleVS | undefined;

  beforeAll(async () => {
    pool = await oracledb.createPool({
      user: env.ORACLE_USERNAME,
      password: env.ORACLE_PASSWORD,
      connectString: env.ORACLE_DSN,
    });

    const pref = {
      provider: "database",
      model: env.DEMO_ONNX_MODEL,
    };
    connection = await pool.getConnection();
    embedder = new OracleEmbeddings(connection, pref);
  });

  beforeEach(async () => {
    // Drop table for the next test.
    await dropTablePurge(connection as oracledb.Connection, tableName);
  });

  afterAll(async () => {
    await dropTablePurge(connection as oracledb.Connection, tableName);
    await connection?.close();
    await pool.close();
  });

  test("Test vectorstore fromDocuments", async () => {
    let connection: oracledb.Connection | undefined;

    try {
      connection = await pool.getConnection();
      const dbConfig: OracleDBVSStoreArgs = {
        client: pool,
        tableName,
        distanceStrategy: DistanceStrategy.DOT_PRODUCT,
        query: "What are salient features of oracledb",
      };

      const docs = [];
      docs.push(new Document({ pageContent: "I like soccer." }));
      docs.push(new Document({ pageContent: "I love Stephen King." }));

      oraclevs = await OracleVS.fromDocuments(docs, embedder, dbConfig);

      await createIndex(connection, oraclevs, {
        idxName: "embeddings_idx",
        idxType: "IVF",
        neighborPart: 64,
        accuracy: 90,
      });

      const embedding = await embedder.embedQuery(
        "What is your favourite sport?"
      );
      const matches = await oraclevs.similaritySearchVectorWithScore(
        embedding,
        1
      );

      expect(matches).toHaveLength(1);
    } finally {
      if (connection) {
        await connection?.close();
      }
    }
  });

  test("Test vectorstore addDocuments", async () => {
    const dbConfig: OracleDBVSStoreArgs = {
      client: pool,
      tableName,
      distanceStrategy: DistanceStrategy.DOT_PRODUCT,
      query: "What are salient features of oracledb",
    };

    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    const docs = [
      new Document({ pageContent: "hello", metadata: { a: 2 } }),
      new Document({ pageContent: "car", metadata: { a: 1 } }),
      new Document({ pageContent: "adjective", metadata: { a: 1 } }),
      new Document({ pageContent: "hi", metadata: { a: 1 } }),
    ];
    await oraclevs.addDocuments(docs);
    const results1 = await oraclevs.similaritySearch("hello!", 1);
    expect(results1).toHaveLength(1);
    expect(results1).toEqual([
      expect.objectContaining({ metadata: { a: 2 }, pageContent: "hello" }),
    ]);

    const dbFilter = { key: "a", oper: "EQ", value: 1 }; // { a:1 }
    const results2 = await oraclevs.similaritySearchWithScore(
      "hello!",
      1,
      dbFilter
    );
    expect(results2).toHaveLength(1);
  });

  test("Test vectorstore addDocuments and find using filter IN Clause", async () => {
    const dbConfig: OracleDBVSStoreArgs = {
      client: pool,
      tableName,
      distanceStrategy: DistanceStrategy.DOT_PRODUCT,
      query: "What are salient features of oracledb",
    };

    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    const makeDoc = (
      content: string,
      author: string | string[],
      category = "research/AI"
    ) => ({
      pageContent: content,
      metadata: {
        category,
        author: Array.isArray(author) ? author : [author],
        tags: ["AI", "ML"],
        status: "release",
      },
    });

    const docs = [
      makeDoc(
        "Alice discusses the application of machine learning and AI research in predicting football match outcomes.",
        ["Alice", "Bob"],
        "sports"
      ),
      makeDoc(
        "Geoffrey Hinton explores the future of deep learning and its impact on AI research.",
        "Geoffrey Hinton"
      ),
      makeDoc(
        "Yoshua Bengio presents breakthroughs in neural network architectures for natural language understanding.",
        "Yoshua Bengio"
      ),
      makeDoc(
        "Andrew Ng shares insights on scaling AI education to democratize access to machine learning tools.",
        "Andrew Ng"
      ),
    ];

    await oraclevs.addDocuments(docs);

    const filter = { author: { IN: ["Andrew Ng", "Demis Hassabis"] } };
    const results = await oraclevs.similaritySearch(
      "latest advances in AI research for education",
      1,
      filter
    );

    expect(results).toHaveLength(1);
    expect(results).toEqual([
      expect.objectContaining({
        metadata: { category: "research/AI", author: ["Andrew Ng"], tags: ["AI", "ML"], status: "release" },
        pageContent:
          "Andrew Ng shares insights on scaling AI education to democratize access to machine learning tools.",
      }),
    ]);
  });

  test("should handle a simple _and clause", async () => {
    const dbConfig: OracleDBVSStoreArgs = {
      client: pool,
      tableName,
      distanceStrategy: DistanceStrategy.DOT_PRODUCT,
      query: "What are salient features of oracledb",
    };
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    // Sample documents
    const docs = [
      new Document({
        pageContent: "A thrilling fantasy novel with dragons and magic.",
        metadata: { category: "books", price: 15 },
      }),
      new Document({
        pageContent: "A guide to healthy cooking with fresh vegetables.",
        metadata: { category: "books", price: 25 },
      }),
      new Document({
        pageContent: "A strategy board game with medieval warfare theme.",
        metadata: { category: "games", price: 40 },
      }),
      new Document({
        pageContent: "A romantic novel set in Paris.",
        metadata: { category: "books", price: 10 },
      }),
    ];
    await oraclevs.addDocuments(docs);
    const filter: Metadata = {
      _and: [
        { key: "category", oper: "EQ", value: "books" },
        { key: "price", oper: "LTE", value: 20 },
      ],
    };
    const results = await oraclevs.similaritySearch("test", 2, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(2);

    results.forEach((doc) => {
      expect(doc.metadata.category).toBe("books");
      expect(doc.metadata.price).toBeLessThanOrEqual(20);
    });
  });

  test("should handle a simple _or clause", async () => {
    const dbConfig: OracleDBVSStoreArgs = {
      client: pool,
      tableName,
      distanceStrategy: DistanceStrategy.DOT_PRODUCT,
      query: "What are salient features of oracledb",
    };
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    // Sample documents
    const docs = [
      new Document({
        pageContent: "A thrilling fantasy novel with dragons and magic.",
        metadata: { category: "books", price: 15 },
      }),
      new Document({
        pageContent: "A guide to healthy cooking with fresh vegetables.",
        metadata: { category: "books", price: 25 },
      }),
      new Document({
        pageContent: "A strategy board game with medieval warfare theme.",
        metadata: { category: "games", price: 15 },
      }),
      new Document({
        pageContent: "A romantic novel set in Paris.",
        metadata: { category: "books", price: 10 },
      }),
      new Document({
        pageContent:
          "A strategy board game with medieval Civil Constructions theme.",
        metadata: { category: "games", price: 40 },
      }),
    ];
    await oraclevs.addDocuments(docs);
    const filter: Metadata = {
      _or: [
        { key: "category", oper: "EQ", value: "books" },
        { key: "price", oper: "LTE", value: 20 },
      ],
    };
    const results = await oraclevs.similaritySearch("test", 6, filter);
    expect(results).toBeInstanceOf(Array);
    expect(results).toHaveLength(4);

    results.forEach((doc) => {
      expect(
        doc.metadata.price <= 20 || doc.metadata.category === "books"
      ).toBe(true);
    });
  });

  test("should handle a nested _and and _or clause", async () => {
    const dbConfig: OracleDBVSStoreArgs = {
      client: pool,
      tableName,
      distanceStrategy: DistanceStrategy.DOT_PRODUCT,
      query: "What are salient features of oracledb",
    };
    oraclevs = new OracleVS(embedder, dbConfig);
    await oraclevs.initialize();

    // Sample docs
    const docs = [
      new Document({
        id: "1",
        pageContent: "A thrilling mystery novel",
        metadata: { category: "books", price: 15, rating: 4.5 },
      }),
      new Document({
        id: "2",
        pageContent: "An expensive historical book",
        metadata: { category: "books", price: 35, rating: 4.7 },
      }),

      new Document({
        id: "3",
        pageContent: "Affordable cooking guide",
        metadata: { category: "books", price: 18, rating: 4.2 },
      }),

      new Document({
        id: "4",
        pageContent: "Wireless Bluetooth headphones",
        metadata: { category: "electronics", price: 50, rating: 4.1 },
      }),

      new Document({
        id: "5",
        pageContent: "Budget wired earphones",
        metadata: { category: "electronics", price: 15, rating: 3.9 },
      }),
    ];

    // Insert into the vector store with ids provided.
    await oraclevs.addDocuments(docs, { ids: ["1", "2", "3", "4", "5"] });

    const filter: Metadata = {
      _or: [
        // First OR branch: simple AND group
        {
          _and: [
            { key: "category", oper: "EQ", value: "books" },
            { key: "price", oper: "LTE", value: 20 },
          ],
        },
        // Second OR branch: nested OR inside AND
        {
          _and: [
            { key: "category", oper: "EQ", value: "electronics" },
            {
              _or: [
                { key: "price", oper: "LTE", value: 20 },
                { key: "rating", oper: "GTE", value: 4.5 },
              ],
            },
          ],
        },
      ],
    };

    // Complex filter example with _or and nested _and/_or
    const results = await oraclevs.similaritySearch("test", 10, filter);
    const expectedId = (id: string) => {
      return String(
        Buffer.from(
          createHash("sha256")
            .update(id)
            .digest("hex")
            .substring(0, 16)
            .toUpperCase(),
          "hex"
        )
      );
    };
    expect(results).toHaveLength(3);
    expect(results).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          pageContent: "A thrilling mystery novel",
          metadata: { category: "books", price: 15, rating: 4.5 },
          id: expectedId("1"),
        }),
        expect.objectContaining({
          pageContent: "Budget wired earphones",
          metadata: { category: "electronics", price: 15, rating: 3.9 },
          id: expectedId("5"),
        }),
        expect.objectContaining({
          pageContent: "Affordable cooking guide",
          metadata: { category: "books", price: 18, rating: 4.2 },
          id: expectedId("3"),
        }),
      ])
    );
  });
});

/*
describe("generateSQLFromFilters - nested _and and _or", () => {
>>>>>>> REPLACE
  it("handles single-level _or group", () => {
    const filters: FilterGroup = {
      _or: [
        { key: "id", oper: "EQ", value: "ba" },
        { key: "id", oper: "EQ", value: "st" }
      ]
    };

    const { sql, binds } = generateSQLFromFilters(filters);
    expect(sql).toBe(
      `(JSON_EXISTS(metadata, '$.id?(@ == :bind0)') OR JSON_EXISTS(metadata, '$.id?(@ == :bind1)'))`
    );
    expect(binds).toEqual({ bind0: "ba", bind1: "st" });
  });

  it("handles nested _or with _and", () => {
    const filters: FilterGroup = {
      _or: [
        { key: "id", oper: "EQ", value: "ba" },
        {
          _and: [
            { key: "order", oper: "LTE", value: 4 },
            { key: "id", oper: "EQ", value: "st" }
          ]
        }
      ]
    };

    const { sql, binds } = generateSQLFromFilters(filters);

    // SQL check
    expect(sql).toBe(
      `(JSON_EXISTS(metadata, '$.id?(@ == :bind0)') OR (JSON_EXISTS(metadata, '$.order?(@ <= :bind1)') AND JSON_EXISTS(metadata, '$.id?(@ == :bind2)')))`
    );

    // Bind values check
    expect(binds).toEqual({
      bind0: "ba",
      bind1: 4,
      bind2: "st"
    });
  });

  it("handles multiple nesting levels", () => {
    const filters: FilterGroup = {
      _and: [
        {
          _or: [
            { key: "id", oper: "EQ", value: "ba" },
            { key: "id", oper: "EQ", value: "st" }
          ]
        },
        {
          _and: [
            { key: "order", oper: "LTE", value: 4 },
            {
              _or: [
                { key: "status", oper: "EQ", value: "active" },
                { key: "status", oper: "EQ", value: "pending" }
              ]
            }
          ]
        }
      ]
    };

    const { sql, binds } = generateSQLFromFilters(filters);

    expect(sql).toBe(
      `((JSON_EXISTS(metadata, '$.id?(@ == :bind0)') OR JSON_EXISTS(metadata, '$.id?(@ == :bind1)')) AND (JSON_EXISTS(metadata, '$.order?(@ <= :bind2)') AND (JSON_EXISTS(metadata, '$.status?(@ == :bind3)') OR JSON_EXISTS(metadata, '$.status?(@ == :bind4)'))))`
    );

    expect(binds).toEqual({
      bind0: "ba",
      bind1: "st",
      bind2: 4,
      bind3: "active",
      bind4: "pending"
    });
  });
});
*/
