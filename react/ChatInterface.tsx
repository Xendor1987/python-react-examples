'use client'

/**
 * Chat Interface Component
 *
 * A full-featured chat interface for AI assistants with:
 * - Custom Markdown renderer (no external dependencies)
 * - Session management (history, create, delete)
 * - Source citations with document preview
 * - Responsive design (desktop sidebar + mobile overlay)
 * - Loading states and error handling
 *
 * Tech stack: React 18, TypeScript, Tailwind CSS, Lucide icons
 */

import { useState, useRef, useEffect, useMemo } from 'react'
import {
  Send,
  Loader2,
  FileText,
  Mail,
  MessageCircle,
  Plus,
  Trash2,
  History,
  X
} from 'lucide-react'

// ============================================================================
// Types
// ============================================================================

interface Source {
  document_id: number
  source_type: 'email' | 'telegram' | 'whatsapp' | 'document'
  subject?: string
  chat_name?: string
  sender?: string
  snippet: string
  date?: string
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

interface ChatSession {
  id: number
  title: string
  updated_at: string
}

interface Document {
  id: number
  source_type: string
  subject?: string
  sender?: string
  recipients?: string
  sent_at?: string
  content: string
}

// ============================================================================
// Custom Markdown Renderer (No External Dependencies)
// ============================================================================

function MarkdownRenderer({ content }: { content: string }) {
  const rendered = useMemo(() => {
    const lines = content.split('\n')
    const elements: JSX.Element[] = []
    let listItems: string[] = []
    let listType: 'ul' | 'ol' | null = null

    // Flush accumulated list items
    const flushList = () => {
      if (listItems.length > 0 && listType) {
        const ListTag = listType
        elements.push(
          <ListTag
            key={elements.length}
            className={listType === 'ol' ? 'list-decimal ml-5 my-2' : 'list-disc ml-5 my-2'}
          >
            {listItems.map((item, i) => (
              <li key={i} className="my-1">{formatInline(item)}</li>
            ))}
          </ListTag>
        )
        listItems = []
        listType = null
      }
    }

    // Format inline elements (bold, italic, code)
    const formatInline = (text: string): (string | JSX.Element)[] => {
      const result: (string | JSX.Element)[] = []
      let remaining = text
      let keyIdx = 0

      while (remaining.length > 0) {
        // Bold text **text** or __text__
        const boldMatch = remaining.match(/^(.*?)\*\*(.+?)\*\*(.*)$/) ||
                         remaining.match(/^(.*?)__(.+?)__(.*)$/)
        if (boldMatch) {
          if (boldMatch[1]) result.push(boldMatch[1])
          result.push(
            <strong key={keyIdx++} className="font-semibold">
              {boldMatch[2]}
            </strong>
          )
          remaining = boldMatch[3]
          continue
        }

        // Italic *text*
        const italicMatch = remaining.match(/^(.*?)\*([^*]+)\*(.*)$/)
        if (italicMatch) {
          if (italicMatch[1]) result.push(italicMatch[1])
          result.push(
            <em key={keyIdx++} className="italic">
              {italicMatch[2]}
            </em>
          )
          remaining = italicMatch[3]
          continue
        }

        // Inline code `code`
        const codeMatch = remaining.match(/^(.*?)`(.+?)`(.*)$/)
        if (codeMatch) {
          if (codeMatch[1]) result.push(codeMatch[1])
          result.push(
            <code
              key={keyIdx++}
              className="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono"
            >
              {codeMatch[2]}
            </code>
          )
          remaining = codeMatch[3]
          continue
        }

        // No matches - add remainder and exit
        result.push(remaining)
        break
      }

      return result
    }

    // Process each line
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]

      // Headers
      const h3Match = line.match(/^###\s+(.+)$/)
      if (h3Match) {
        flushList()
        elements.push(
          <h3 key={elements.length} className="font-semibold text-base mt-3 mb-1">
            {formatInline(h3Match[1])}
          </h3>
        )
        continue
      }

      const h2Match = line.match(/^##\s+(.+)$/)
      if (h2Match) {
        flushList()
        elements.push(
          <h2 key={elements.length} className="font-semibold text-lg mt-3 mb-1">
            {formatInline(h2Match[1])}
          </h2>
        )
        continue
      }

      const h1Match = line.match(/^#\s+(.+)$/)
      if (h1Match) {
        flushList()
        elements.push(
          <h1 key={elements.length} className="font-bold text-xl mt-3 mb-1">
            {formatInline(h1Match[1])}
          </h1>
        )
        continue
      }

      // Ordered list
      const olMatch = line.match(/^(\d+)\.\s+(.+)$/)
      if (olMatch) {
        if (listType !== 'ol') {
          flushList()
          listType = 'ol'
        }
        listItems.push(olMatch[2])
        continue
      }

      // Unordered list
      const ulMatch = line.match(/^[-*]\s+(.+)$/)
      if (ulMatch) {
        if (listType !== 'ul') {
          flushList()
          listType = 'ul'
        }
        listItems.push(ulMatch[1])
        continue
      }

      // Empty line
      if (line.trim() === '') {
        flushList()
        elements.push(<br key={elements.length} />)
        continue
      }

      // Regular text
      flushList()
      elements.push(
        <p key={elements.length} className="my-1">
          {formatInline(line)}
        </p>
      )
    }

    flushList()
    return elements
  }, [content])

  return <div className="markdown-content">{rendered}</div>
}

// ============================================================================
// Sub-components
// ============================================================================

function SourceIcon({ type }: { type: string }) {
  switch (type) {
    case 'email':
      return <Mail size={14} className="text-blue-500" />
    case 'telegram':
      return <MessageCircle size={14} className="text-sky-500" />
    default:
      return <FileText size={14} className="text-gray-500" />
  }
}

interface SourceCardProps {
  source: Source
  onClick?: () => void
}

function SourceCard({ source, onClick }: SourceCardProps) {
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return ''
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    })
  }

  return (
    <div
      className={`
        p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm
        ${onClick ? 'cursor-pointer hover:bg-gray-100 transition-colors' : ''}
      `}
      onClick={onClick}
    >
      <div className="flex items-center gap-2 mb-2">
        <SourceIcon type={source.source_type} />
        <span className="font-medium text-gray-900">
          {source.source_type === 'email' ? source.subject : source.chat_name}
        </span>
      </div>
      <p className="text-gray-600 text-xs mb-2 line-clamp-2">{source.snippet}</p>
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>
          {source.source_type === 'email' ? source.sender : source.chat_name}
        </span>
        <span>{formatDate(source.date)}</span>
      </div>
    </div>
  )
}

interface DocumentModalProps {
  document: Document
  onClose: () => void
}

function DocumentModal({ document, onClose }: DocumentModalProps) {
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return ''
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <SourceIcon type={document.source_type} />
            <h2 className="font-semibold text-gray-900">
              {document.subject || 'Message'}
            </h2>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded">
            <X size={20} className="text-gray-500" />
          </button>
        </div>

        {/* Metadata */}
        <div className="p-4 space-y-3 text-sm border-b border-gray-100">
          {document.sender && (
            <div className="flex">
              <span className="text-gray-500 w-24">From:</span>
              <span className="text-gray-900">{document.sender}</span>
            </div>
          )}
          {document.recipients && (
            <div className="flex">
              <span className="text-gray-500 w-24">To:</span>
              <span className="text-gray-900">{document.recipients}</span>
            </div>
          )}
          {document.sent_at && (
            <div className="flex">
              <span className="text-gray-500 w-24">Date:</span>
              <span className="text-gray-900">{formatDate(document.sent_at)}</span>
            </div>
          )}
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto flex-1">
          <p className="whitespace-pre-wrap text-gray-800">{document.content}</p>
        </div>
      </div>
    </div>
  )
}

interface MessageBubbleProps {
  message: ChatMessage
  onSourceClick?: (documentId: number) => void
}

function MessageBubble({ message, onSourceClick }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] ${isUser ? 'order-1' : 'order-2'}`}>
        <div
          className={`
            px-4 py-3 rounded-2xl
            ${isUser
              ? 'bg-blue-600 text-white rounded-br-md'
              : 'bg-white border border-gray-200 text-gray-900 rounded-bl-md'
            }
          `}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <MarkdownRenderer content={message.content} />
          )}
        </div>

        {/* Source citations */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-xs font-medium text-gray-500 px-1">
              Sources ({message.sources.length}):
            </p>
            <div className="grid gap-2">
              {message.sources.slice(0, 3).map((source, idx) => (
                <SourceCard
                  key={idx}
                  source={source}
                  onClick={
                    source.document_id
                      ? () => onSourceClick?.(source.document_id)
                      : undefined
                  }
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

interface SessionListProps {
  sessions: ChatSession[]
  currentSessionId?: number
  onSelectSession: (sessionId: number) => void
  onNewSession: () => void
  onDeleteSession: (sessionId: number) => void
  onClose: () => void
}

function SessionList({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onClose
}: SessionListProps) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffDays = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24)
    )

    if (diffDays === 0) return 'Today'
    if (diffDays === 1) return 'Yesterday'
    return date.toLocaleDateString('en-US', { day: 'numeric', month: 'short' })
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between bg-white">
        <h2 className="font-semibold text-gray-900">Chat History</h2>
        <button
          onClick={onClose}
          className="md:hidden p-1 hover:bg-gray-100 rounded"
        >
          <X size={20} className="text-gray-500" />
        </button>
      </div>

      {/* New chat button */}
      <button
        onClick={onNewSession}
        className="m-4 flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
      >
        <Plus size={18} />
        New Chat
      </button>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <p className="p-4 text-sm text-gray-500 text-center">
            No saved chats
          </p>
        ) : (
          <div className="space-y-1 p-2">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`
                  group flex items-center gap-2 p-3 rounded-lg cursor-pointer transition-colors
                  ${currentSessionId === session.id
                    ? 'bg-blue-50 text-blue-700'
                    : 'hover:bg-gray-100'
                  }
                `}
                onClick={() => onSelectSession(session.id)}
              >
                <MessageCircle size={16} className="shrink-0 text-gray-400" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">
                    {session.title || 'New Chat'}
                  </p>
                  <p className="text-xs text-gray-400">
                    {formatDate(session.updated_at)}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteSession(session.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-opacity"
                >
                  <Trash2 size={14} className="text-red-500" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================================
// Mock API Functions (Replace with real API calls)
// ============================================================================

async function getChatSessions(): Promise<ChatSession[]> {
  // Replace with actual API call
  return []
}

async function getChatSession(id: number): Promise<{ messages: ChatMessage[] }> {
  // Replace with actual API call
  return { messages: [] }
}

async function deleteChatSession(id: number): Promise<void> {
  // Replace with actual API call
}

async function sendMessage(
  message: string,
  sessionId?: number
): Promise<{ answer: string; sources: Source[]; session_id: number }> {
  // Replace with actual API call
  return {
    answer: 'This is a mock response.',
    sources: [],
    session_id: sessionId || 1
  }
}

async function getDocument(id: number): Promise<Document> {
  // Replace with actual API call
  return {
    id,
    source_type: 'email',
    subject: 'Sample Document',
    sender: 'sender@example.com',
    content: 'Document content here...'
  }
}

// ============================================================================
// Main Component
// ============================================================================

export default function ChatInterface() {
  // State
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<number | undefined>()
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [isLoadingDocument, setIsLoadingDocument] = useState(false)

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  // Handlers
  const loadSessions = async () => {
    try {
      const data = await getChatSessions()
      setSessions(data)
    } catch (err) {
      console.error('Failed to load sessions:', err)
    }
  }

  const loadSession = async (id: number) => {
    try {
      const data = await getChatSession(id)
      setMessages(data.messages)
      setSessionId(id)
      setShowHistory(false)
    } catch (err) {
      console.error('Failed to load session:', err)
      setError('Failed to load chat')
    }
  }

  const handleNewSession = () => {
    setMessages([])
    setSessionId(undefined)
    setError(null)
    setShowHistory(false)
  }

  const handleDeleteSession = async (id: number) => {
    try {
      await deleteChatSession(id)
      setSessions((prev) => prev.filter((s) => s.id !== id))
      if (sessionId === id) {
        handleNewSession()
      }
    } catch (err) {
      console.error('Failed to delete session:', err)
    }
  }

  const handleSourceClick = async (documentId: number) => {
    setIsLoadingDocument(true)
    try {
      const doc = await getDocument(documentId)
      setSelectedDocument(doc)
    } catch (err) {
      console.error('Failed to load document:', err)
    } finally {
      setIsLoadingDocument(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = { role: 'user', content: input.trim() }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setError(null)
    setIsLoading(true)

    try {
      const response = await sendMessage(input.trim(), sessionId)

      if (!sessionId && response.session_id) {
        setSessionId(response.session_id)
        await loadSessions()
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      setError('Failed to get response. Please check your connection.')
      console.error('Chat error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  // Quick action buttons for empty state
  const quickActions = [
    'What emails came in this week?',
    'What did we agree with partners?',
    'Are there any open issues?'
  ]

  return (
    <div className="flex h-full">
      {/* Desktop sidebar */}
      <div className="hidden md:block w-72 border-r border-gray-200">
        <SessionList
          sessions={sessions}
          currentSessionId={sessionId}
          onSelectSession={loadSession}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
          onClose={() => setShowHistory(false)}
        />
      </div>

      {/* Mobile history overlay */}
      {showHistory && (
        <div
          className="md:hidden fixed inset-0 bg-black/50 z-50"
          onClick={() => setShowHistory(false)}
        >
          <div
            className="absolute left-0 top-0 h-full w-72 bg-white"
            onClick={(e) => e.stopPropagation()}
          >
            <SessionList
              sessions={sessions}
              currentSessionId={sessionId}
              onSelectSession={loadSession}
              onNewSession={handleNewSession}
              onDeleteSession={handleDeleteSession}
              onClose={() => setShowHistory(false)}
            />
          </div>
        </div>
      )}

      {/* Main chat area */}
      <div className="flex-1 flex flex-col h-full min-w-0">
        {/* Mobile header */}
        <div className="md:hidden flex items-center justify-between p-4 border-b border-gray-200 bg-white">
          <button
            onClick={() => setShowHistory(true)}
            className="p-2 hover:bg-gray-100 rounded-lg"
          >
            <History size={20} className="text-gray-600" />
          </button>
          <h1 className="font-semibold text-gray-900">AI Assistant</h1>
          <button
            onClick={handleNewSession}
            className="p-2 hover:bg-gray-100 rounded-lg"
          >
            <Plus size={20} className="text-gray-600" />
          </button>
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            // Empty state
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                <MessageCircle size={32} className="text-blue-600" />
              </div>
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                AI Assistant Chat
              </h2>
              <p className="text-gray-500 max-w-md">
                Ask questions about your documents, projects, or communications.
                The AI will analyze your data and provide answers with sources.
              </p>
              <div className="mt-6 grid gap-2 text-sm">
                {quickActions.map((action, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInput(action)}
                    className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-gray-700 transition-colors"
                  >
                    {action}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, idx) => (
                <MessageBubble
                  key={idx}
                  message={message}
                  onSourceClick={handleSourceClick}
                />
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="px-4 py-3 bg-white border border-gray-200 rounded-2xl rounded-bl-md">
                    <Loader2 size={20} className="animate-spin text-gray-400" />
                  </div>
                </div>
              )}
              {error && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                  {error}
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input area */}
        <div className="border-t border-gray-200 p-4 bg-white">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your question..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-4 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={20} />
            </button>
          </form>
        </div>
      </div>

      {/* Document modal */}
      {selectedDocument && (
        <DocumentModal
          document={selectedDocument}
          onClose={() => setSelectedDocument(null)}
        />
      )}

      {/* Loading overlay for document */}
      {isLoadingDocument && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
          <Loader2 size={32} className="animate-spin text-white" />
        </div>
      )}
    </div>
  )
}
