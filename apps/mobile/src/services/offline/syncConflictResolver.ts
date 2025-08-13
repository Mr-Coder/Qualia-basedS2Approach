export type ConflictResolutionStrategy = 'client_wins' | 'server_wins' | 'last_write_wins' | 'merge';

export interface SyncConflict<T> {
  type: string;
  clientData: T;
  serverData: T;
  clientTimestamp: number;
  serverTimestamp: number;
  metadata?: Record<string, any>;
}

export interface ConflictResolution<T> {
  resolved: T;
  strategy: ConflictResolutionStrategy;
  conflicts: string[];
}

export class SyncConflictResolver {
  private static instance: SyncConflictResolver;

  static getInstance(): SyncConflictResolver {
    if (!SyncConflictResolver.instance) {
      SyncConflictResolver.instance = new SyncConflictResolver();
    }
    return SyncConflictResolver.instance;
  }

  resolve<T>(
    conflict: SyncConflict<T>,
    strategy: ConflictResolutionStrategy = 'last_write_wins'
  ): ConflictResolution<T> {
    switch (strategy) {
      case 'client_wins':
        return this.clientWins(conflict);
      case 'server_wins':
        return this.serverWins(conflict);
      case 'last_write_wins':
        return this.lastWriteWins(conflict);
      case 'merge':
        return this.merge(conflict);
      default:
        return this.lastWriteWins(conflict);
    }
  }

  private clientWins<T>(conflict: SyncConflict<T>): ConflictResolution<T> {
    return {
      resolved: conflict.clientData,
      strategy: 'client_wins',
      conflicts: this.detectConflicts(conflict.clientData, conflict.serverData),
    };
  }

  private serverWins<T>(conflict: SyncConflict<T>): ConflictResolution<T> {
    return {
      resolved: conflict.serverData,
      strategy: 'server_wins',
      conflicts: this.detectConflicts(conflict.clientData, conflict.serverData),
    };
  }

  private lastWriteWins<T>(conflict: SyncConflict<T>): ConflictResolution<T> {
    const resolved = conflict.clientTimestamp > conflict.serverTimestamp
      ? conflict.clientData
      : conflict.serverData;

    return {
      resolved,
      strategy: 'last_write_wins',
      conflicts: this.detectConflicts(conflict.clientData, conflict.serverData),
    };
  }

  private merge<T>(conflict: SyncConflict<T>): ConflictResolution<T> {
    // Type-specific merge strategies
    if (conflict.type === 'message') {
      return this.mergeMessages(conflict);
    } else if (conflict.type === 'whiteboard') {
      return this.mergeWhiteboard(conflict);
    } else if (conflict.type === 'user_state') {
      return this.mergeUserState(conflict);
    }

    // Default: last write wins
    return this.lastWriteWins(conflict);
  }

  private mergeMessages<T>(conflict: SyncConflict<T>): ConflictResolution<T> {
    // Messages are immutable - use last write wins
    return this.lastWriteWins(conflict);
  }

  private mergeWhiteboard<T>(conflict: SyncConflict<T>): ConflictResolution<T> {
    const clientData = conflict.clientData as any;
    const serverData = conflict.serverData as any;

    // Merge whiteboard paths - combine unique paths from both
    if (clientData.paths && serverData.paths) {
      const pathMap = new Map();

      // Add server paths first
      serverData.paths.forEach((path: any) => {
        pathMap.set(path.id, path);
      });

      // Add client paths (will override if same ID)
      clientData.paths.forEach((path: any) => {
        pathMap.set(path.id, path);
      });

      const merged = {
        ...serverData,
        ...clientData,
        paths: Array.from(pathMap.values()).sort((a: any, b: any) => 
          a.timestamp - b.timestamp
        ),
      };

      return {
        resolved: merged as T,
        strategy: 'merge',
        conflicts: [],
      };
    }

    return this.lastWriteWins(conflict);
  }

  private mergeUserState<T>(conflict: SyncConflict<T>): ConflictResolution<T> {
    const clientData = conflict.clientData as any;
    const serverData = conflict.serverData as any;

    // Merge user preferences
    const merged = {
      ...serverData,
      ...clientData,
      // Server wins for critical fields
      userId: serverData.userId || clientData.userId,
      email: serverData.email || clientData.email,
      // Client wins for preferences
      preferences: {
        ...(serverData.preferences || {}),
        ...(clientData.preferences || {}),
      },
      // Merge arrays
      rooms: Array.from(new Set([
        ...(serverData.rooms || []),
        ...(clientData.rooms || []),
      ])),
      // Last update timestamp
      lastUpdated: Math.max(
        serverData.lastUpdated || 0,
        clientData.lastUpdated || 0
      ),
    };

    return {
      resolved: merged as T,
      strategy: 'merge',
      conflicts: this.detectConflicts(clientData, serverData),
    };
  }

  private detectConflicts(clientData: any, serverData: any): string[] {
    const conflicts: string[] = [];

    const compareObjects = (obj1: any, obj2: any, path = ''): void => {
      if (obj1 === obj2) return;

      if (typeof obj1 !== typeof obj2) {
        conflicts.push(path || 'root');
        return;
      }

      if (typeof obj1 !== 'object' || obj1 === null) {
        if (obj1 !== obj2) {
          conflicts.push(path || 'root');
        }
        return;
      }

      // Compare arrays
      if (Array.isArray(obj1) && Array.isArray(obj2)) {
        if (obj1.length !== obj2.length) {
          conflicts.push(path || 'root');
          return;
        }
        // Simple array comparison - could be enhanced
        for (let i = 0; i < obj1.length; i++) {
          compareObjects(obj1[i], obj2[i], `${path}[${i}]`);
        }
        return;
      }

      // Compare objects
      const keys = new Set([...Object.keys(obj1), ...Object.keys(obj2)]);
      for (const key of keys) {
        const newPath = path ? `${path}.${key}` : key;
        if (!(key in obj1) || !(key in obj2)) {
          conflicts.push(newPath);
        } else {
          compareObjects(obj1[key], obj2[key], newPath);
        }
      }
    };

    compareObjects(clientData, serverData);
    return conflicts;
  }

  // Batch conflict resolution
  resolveBatch<T>(
    conflicts: SyncConflict<T>[],
    strategy: ConflictResolutionStrategy = 'last_write_wins'
  ): ConflictResolution<T>[] {
    return conflicts.map(conflict => this.resolve(conflict, strategy));
  }
}