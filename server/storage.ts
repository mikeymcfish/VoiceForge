// This app uses in-memory storage for processing state
// No persistent storage needed as it's a text preprocessing tool

export interface IStorage {
  // Storage interface - currently no persistence needed
}

export class MemStorage implements IStorage {
  constructor() {}
}

export const storage = new MemStorage();
