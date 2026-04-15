"use client";

import React, { useState } from 'react';
import { Send, User, Search, RefreshCw, BarChart2 } from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  citations?: string[];
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMsg,
          history: messages,
        }),
      });

      if (!response.ok) {
        throw new Error('API Error');
      }

      const data = await response.json();
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        citations: data.citations
      }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: "Sorry, I encountered an error. Please try again." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full">
      {/* Sidebar */}
      <aside className="w-64 border-r border-[#e94560]/20 bg-[#0f0f1e]/95 flex flex-col hidden md:flex">
        <div className="p-4 border-b border-[#e94560]/20">
          <h2 className="text-[#e94560] font-bold text-xl flex items-center gap-2">
            <BarChart2 className="w-5 h-5"/> MF FAQ Assistant
          </h2>
          <div className="text-xs text-white/60 mt-2 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_6px_rgba(39,174,96,0.5)]"></span>
            System Ready
          </div>
        </div>
        <div className="p-4 flex-1 overflow-y-auto">
          <button 
            onClick={() => setMessages([])} 
            className="w-full py-2 px-4 rounded-lg bg-[#e94560] hover:bg-[#d63d56] transition-colors text-white font-medium flex justify-center items-center gap-2"
          >
            <RefreshCw className="w-4 h-4"/> New Chat
          </button>
        </div>
        <div className="p-4 border-t border-white/5 text-center text-xs text-white/30">
          Facts-only RAG System<br/>
          15 HDFC Schemes from Groww<br/><br/>
          ⚠️ Not financial advice
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Messages List */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 pb-32">
          {messages.length === 0 ? (
            <div className="max-w-3xl mx-auto mt-10">
              <div className="welcome-banner">
                  <h1>📊 Mutual Fund FAQ Assistant</h1>
                  <p className="text-white/80 text-lg mb-2">
                      Get instant, fact-based answers about 15 HDFC Mutual Fund schemes
                  </p>
                  <p className="disclaimer">
                      ⚠️ Facts-only — No investment advice. Data sourced from Groww.
                  </p>
              </div>
              <div className="mt-8">
                <h3 className="text-center text-white/60 mb-4 font-medium">💡 Try asking:</h3>
                <div className="flex flex-wrap gap-3 justify-center">
                  {["What is the expense ratio of HDFC Mid-Cap Fund?", "What is the minimum SIP amount for HDFC ELSS?", "Who is the fund manager of HDFC Mid-Cap Fund?"].map((q, i) => (
                    <button 
                      key={i}
                      onClick={() => setInput(q)}
                      className="py-2 px-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-[#e94560]/50 transition-all text-sm text-left max-w-[300px]"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-4">
              {messages.map((msg, i) => (
                <div key={i} className={`flex gap-4 p-5 rounded-xl border border-white/5 ${msg.role === 'user' ? 'bg-white/[0.03]' : 'bg-[#e94560]/[0.05]'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-blue-500/20 text-blue-400' : 'bg-[#e94560]/20 text-[#e94560]'}`}>
                    {msg.role === 'user' ? <User size={18} /> : <BarChart2 size={18} />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-[15px] leading-relaxed whitespace-pre-wrap">{msg.content}</div>
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {msg.citations.map((url, idx) => {
                          try {
                            const domain = new URL(url).hostname;
                            return (
                              <a href={url} target="_blank" rel="noreferrer" key={idx} className="inline-flex items-center gap-1 text-[13px] text-[#e94560] bg-[#e94560]/10 hover:bg-[#e94560]/20 px-2.5 py-1 rounded-md transition-colors border border-[#e94560]/20">
                                🔗 {domain}
                              </a>
                            );
                          } catch { return null; }
                        })}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex gap-4 p-5 rounded-xl border border-white/5 bg-[#e94560]/[0.05]">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-[#e94560]/20 text-[#e94560]">
                    <Search className="w-4 h-4 animate-pulse" />
                  </div>
                  <div className="flex-1">
                    <p className="text-[#e94560] text-sm animate-pulse">Thinking... analyzing query intent & searching facts...</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input Form */}
        <div className="absolute bottom-0 left-0 right-0 p-4 md:p-8 bg-gradient-to-t from-[#0a0a1a] via-[#0a0a1a]/90 to-transparent">
          <div className="max-w-3xl mx-auto relative">
            <form onSubmit={sendMessage} className="relative flex items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about HDFC mutual fund schemes..."
                className="w-full px-6 py-4 pr-14 chat-input font-medium text-[15px]"
                disabled={isLoading}
              />
              <button 
                type="submit" 
                disabled={isLoading || !input.trim()}
                className="absolute right-3 w-10 h-10 rounded-full flex items-center justify-center bg-[#e94560] text-white disabled:opacity-50 hover:bg-[#d63d56] transition-colors"
              >
                <Send size={18} className="translate-x-[-1px] translate-y-[1px]" />
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}
